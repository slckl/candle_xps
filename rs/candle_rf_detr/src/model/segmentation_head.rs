//! RF-DETR Segmentation Head
//!
//! This module implements the segmentation head for RF-DETR instance segmentation.
//! It produces per-query mask logits from spatial features and query embeddings.

use candle_core::{Module, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

/// Configuration for the segmentation head
#[derive(Debug, Clone)]
pub struct SegmentationHeadConfig {
    /// Input dimension (hidden_dim from transformer)
    pub in_dim: usize,
    /// Number of DepthwiseConvBlocks
    pub num_blocks: usize,
    /// Bottleneck ratio for projection (1 means interaction_dim = in_dim)
    pub bottleneck_ratio: usize,
    /// Downsample ratio for mask resolution
    pub downsample_ratio: usize,
}

impl SegmentationHeadConfig {
    pub fn new(in_dim: usize, num_blocks: usize) -> Self {
        Self {
            in_dim,
            num_blocks,
            bottleneck_ratio: 1,
            downsample_ratio: 4,
        }
    }

    /// Get the interaction dimension after bottleneck
    pub fn interaction_dim(&self) -> usize {
        self.in_dim / self.bottleneck_ratio
    }
}

/// Simplified ConvNeXt block without the MLP subnet
/// Implements depthwise conv -> LayerNorm -> pointwise conv -> GELU -> residual
pub struct DepthwiseConvBlock {
    /// Depthwise convolution (groups = channels)
    dwconv: Conv2d,
    /// Layer normalization
    norm: LayerNorm,
    /// Pointwise convolution implemented as linear
    pwconv1: Linear,
    /// Dimension for reshaping
    dim: usize,
}

impl DepthwiseConvBlock {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        // Depthwise conv: groups = dim, so each channel is convolved independently
        let dwconv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: dim,
            ..Default::default()
        };
        let dwconv = conv2d(dim, dim, 3, dwconv_config, vb.pp("dwconv"))?;

        // Layer norm
        let norm = layer_norm(dim, 1e-6, vb.pp("norm"))?;

        // Pointwise conv as linear
        let pwconv1 = linear(dim, dim, vb.pp("pwconv1"))?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();

        // Depthwise conv: (N, C, H, W)
        let x = self.dwconv.forward(x)?;

        // Permute: (N, C, H, W) -> (N, H, W, C)
        let x = x.permute((0, 2, 3, 1))?;

        // LayerNorm on last dimension
        let x = self.norm.forward(&x)?;

        // Pointwise conv (linear)
        let x = self.pwconv1.forward(&x)?;

        // GELU activation
        let x = x.gelu_erf()?;

        // Permute back: (N, H, W, C) -> (N, C, H, W)
        let x = x.permute((0, 3, 1, 2))?;

        // Residual connection
        x.add(&input)
    }
}

/// MLP Block for query feature processing
/// Implements: LayerNorm -> Linear -> GELU -> Linear -> residual
pub struct MLPBlock {
    /// Input layer norm
    norm_in: LayerNorm,
    /// First linear layer (dim -> dim*4)
    fc1: Linear,
    /// Second linear layer (dim*4 -> dim)
    fc2: Linear,
}

impl MLPBlock {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let norm_in = layer_norm(dim, 1e-6, vb.pp("norm_in"))?;
        let fc1 = linear(dim, dim * 4, vb.pp("layers.0"))?;
        let fc2 = linear(dim * 4, dim, vb.pp("layers.2"))?;

        Ok(Self { norm_in, fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();

        // LayerNorm
        let x = self.norm_in.forward(x)?;

        // Linear -> GELU -> Linear
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.fc2.forward(&x)?;

        // Residual
        x.add(&input)
    }
}

/// Segmentation head that produces per-query mask logits
pub struct SegmentationHead {
    /// Configuration
    config: SegmentationHeadConfig,
    /// DepthwiseConvBlocks for spatial feature processing
    blocks: Vec<DepthwiseConvBlock>,
    /// Projection for spatial features (1x1 conv)
    spatial_features_proj: Conv2d,
    /// MLP block for query features
    query_features_block: MLPBlock,
    /// Linear projection for query features
    query_features_proj: Linear,
    /// Learnable bias for mask logits
    bias: Tensor,
}

impl SegmentationHead {
    pub fn load(vb: VarBuilder, config: &SegmentationHeadConfig) -> Result<Self> {
        // Load blocks
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for i in 0..config.num_blocks {
            let block = DepthwiseConvBlock::load(vb.pp(format!("blocks.{}", i)), config.in_dim)?;
            blocks.push(block);
        }

        // Spatial features projection (1x1 conv)
        let interaction_dim = config.interaction_dim();
        let proj_config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let spatial_features_proj = conv2d(
            config.in_dim,
            interaction_dim,
            1,
            proj_config,
            vb.pp("spatial_features_proj"),
        )?;

        // Query features block and projection
        let query_features_block = MLPBlock::load(vb.pp("query_features_block"), config.in_dim)?;
        let query_features_proj =
            linear(config.in_dim, interaction_dim, vb.pp("query_features_proj"))?;

        // Bias
        let bias = vb.get(1, "bias")?;

        Ok(Self {
            config: config.clone(),
            blocks,
            spatial_features_proj,
            query_features_block,
            query_features_proj,
            bias,
        })
    }

    /// Forward pass for inference (export mode)
    ///
    /// # Arguments
    /// * `spatial_features` - Feature map from backbone, shape [B, C, H, W]
    /// * `query_features` - Query embeddings from decoder, shape [B, N, C]
    /// * `image_height` - Original image height (for computing target size)
    /// * `image_width` - Original image width (for computing target size)
    ///
    /// # Returns
    /// Mask logits of shape [B, N, H', W'] where H' = image_height / downsample_ratio
    pub fn forward(
        &self,
        spatial_features: &Tensor,
        query_features: &Tensor,
        image_height: usize,
        image_width: usize,
    ) -> Result<Tensor> {
        // Compute target size for mask resolution
        let target_h = image_height / self.config.downsample_ratio;
        let target_w = image_width / self.config.downsample_ratio;

        // Resize spatial features to target size
        // spatial_features: [B, C, H, W] -> [B, C, target_h, target_w]
        let spatial_features = spatial_features.upsample_nearest2d(target_h, target_w)?;

        // Process through blocks
        let mut spatial_features = spatial_features;
        for block in &self.blocks {
            spatial_features = block.forward(&spatial_features)?;
        }

        // Project spatial features: [B, C, H, W] -> [B, interaction_dim, H, W]
        let spatial_features_proj = self.spatial_features_proj.forward(&spatial_features)?;

        // Process query features: [B, N, C] -> [B, N, interaction_dim]
        let qf = self.query_features_block.forward(query_features)?;
        let qf = self.query_features_proj.forward(&qf)?;

        // Compute mask logits via einsum: 'bchw,bnc->bnhw'
        // spatial_features_proj: [B, C, H, W]
        // qf: [B, N, C]
        // result: [B, N, H, W]
        let mask_logits = self.einsum_bchw_bnc_to_bnhw(&spatial_features_proj, &qf)?;

        // Add bias
        let bias = self.bias.reshape((1, 1, 1, 1))?;
        mask_logits.broadcast_add(&bias)
    }

    /// Compute einsum 'bchw,bnc->bnhw'
    /// This is equivalent to: for each spatial position (h, w), compute dot product
    /// between the C-dimensional feature vector and each of the N query vectors.
    fn einsum_bchw_bnc_to_bnhw(&self, spatial: &Tensor, queries: &Tensor) -> Result<Tensor> {
        // spatial: [B, C, H, W]
        // queries: [B, N, C]
        // output: [B, N, H, W]

        let (b, c, h, w) = spatial.dims4()?;
        let (b2, n, c2) = queries.dims3()?;
        assert_eq!(b, b2, "Batch sizes must match");
        assert_eq!(c, c2, "Channel dimensions must match");

        // Reshape spatial: [B, C, H, W] -> [B, C, H*W]
        let spatial_flat = spatial.reshape((b, c, h * w))?;

        // Transpose spatial: [B, C, H*W] -> [B, H*W, C]
        let spatial_flat = spatial_flat.permute((0, 2, 1))?;

        // queries: [B, N, C]
        // queries transposed: [B, C, N]
        let queries_t = queries.permute((0, 2, 1))?;

        // Matrix multiply: [B, H*W, C] @ [B, C, N] -> [B, H*W, N]
        let result = spatial_flat.matmul(&queries_t)?;

        // Transpose: [B, H*W, N] -> [B, N, H*W]
        let result = result.permute((0, 2, 1))?;

        // Reshape: [B, N, H*W] -> [B, N, H, W]
        result.reshape((b, n, h, w))
    }
}
