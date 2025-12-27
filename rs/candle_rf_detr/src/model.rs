//! RF-DETR Model implementation in Candle
//!
//! This implements the RF-DETR (Roboflow DETR) object detection model.
//! The architecture consists of:
//! - DINOv2 backbone with windowed attention
//! - Multi-scale projector
//! - DETR-style transformer decoder with deformable attention
//! - Detection heads (class + bbox)

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_core as candle;
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder};

// ============================================================================
// Configuration
// ============================================================================

/// RF-DETR model configuration
#[derive(Clone, Debug)]
pub struct RFDETRConfig {
    /// Hidden dimension (256 for base/small/medium, 384 for large)
    pub hidden_dim: usize,
    /// Number of classes (91 for COCO, includes background)
    pub num_classes: usize,
    /// Number of queries (detection slots)
    pub num_queries: usize,
    /// Number of decoder layers
    pub dec_layers: usize,
    /// Patch size for DINOv2 backbone
    pub patch_size: usize,
    /// Number of windows for windowed attention
    #[allow(dead_code)]
    pub num_windows: usize,
    /// Input resolution
    pub resolution: usize,
    /// Self-attention heads
    pub sa_nheads: usize,
    /// Cross-attention heads
    pub ca_nheads: usize,
    /// Deformable attention points
    #[allow(dead_code)]
    pub dec_n_points: usize,
    /// Feature output indexes from backbone
    pub out_feature_indexes: Vec<usize>,
    /// Projector scales (e.g., ["P3", "P4", "P5"])
    pub projector_scales: Vec<String>,
    /// Use bbox reparameterization
    pub bbox_reparam: bool,
    /// Use lite refpoint refinement
    #[allow(dead_code)]
    pub lite_refpoint_refine: bool,
    /// Use two-stage detection
    pub two_stage: bool,
    /// Feedforward dimension
    pub dim_feedforward: usize,
    /// Positional encoding size
    pub positional_encoding_size: usize,
    /// DINOv2 encoder embed dim
    pub encoder_embed_dim: usize,
}

impl RFDETRConfig {
    /// Configuration for RF-DETR Base model
    pub fn base() -> Self {
        Self {
            hidden_dim: 256,
            num_classes: 91,
            num_queries: 300,
            dec_layers: 3,
            patch_size: 14,
            num_windows: 4,
            resolution: 560,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            out_feature_indexes: vec![2, 5, 8, 11],
            projector_scales: vec!["P4".to_string()],
            bbox_reparam: true,
            lite_refpoint_refine: true,
            two_stage: true,
            dim_feedforward: 2048,
            positional_encoding_size: 37,
            encoder_embed_dim: 384,
        }
    }

    /// Configuration for RF-DETR Small model
    pub fn small() -> Self {
        Self {
            hidden_dim: 256,
            num_classes: 91,
            num_queries: 300,
            dec_layers: 3,
            patch_size: 16,
            num_windows: 2,
            resolution: 512,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            out_feature_indexes: vec![3, 6, 9, 12],
            projector_scales: vec!["P4".to_string()],
            bbox_reparam: true,
            lite_refpoint_refine: true,
            two_stage: true,
            dim_feedforward: 2048,
            positional_encoding_size: 32,
            encoder_embed_dim: 384,
        }
    }

    /// Configuration for RF-DETR Medium model
    pub fn medium() -> Self {
        Self {
            hidden_dim: 256,
            num_classes: 91,
            num_queries: 300,
            dec_layers: 4,
            patch_size: 16,
            num_windows: 2,
            resolution: 576,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            out_feature_indexes: vec![3, 6, 9, 12],
            projector_scales: vec!["P4".to_string()],
            bbox_reparam: true,
            lite_refpoint_refine: true,
            two_stage: true,
            dim_feedforward: 2048,
            positional_encoding_size: 36,
            encoder_embed_dim: 384,
        }
    }

    /// Configuration for RF-DETR Large model
    pub fn large() -> Self {
        Self {
            hidden_dim: 384,
            num_classes: 91,
            num_queries: 300,
            dec_layers: 3,
            patch_size: 14,
            num_windows: 4,
            resolution: 560,
            sa_nheads: 12,
            ca_nheads: 24,
            dec_n_points: 4,
            out_feature_indexes: vec![2, 5, 8, 11],
            projector_scales: vec!["P3".to_string(), "P5".to_string()],
            bbox_reparam: true,
            lite_refpoint_refine: true,
            two_stage: true,
            dim_feedforward: 2048,
            positional_encoding_size: 37,
            encoder_embed_dim: 768,
        }
    }

    /// Configuration for RF-DETR Nano model
    pub fn nano() -> Self {
        Self {
            hidden_dim: 256,
            num_classes: 91,
            num_queries: 300,
            dec_layers: 2,
            patch_size: 16,
            num_windows: 2,
            resolution: 384,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            out_feature_indexes: vec![3, 6, 9, 12],
            projector_scales: vec!["P4".to_string()],
            bbox_reparam: true,
            lite_refpoint_refine: true,
            two_stage: true,
            dim_feedforward: 2048,
            positional_encoding_size: 24,
            encoder_embed_dim: 384,
        }
    }

    /// Get number of feature levels
    #[allow(dead_code)]
    pub fn num_feature_levels(&self) -> usize {
        self.projector_scales.len()
    }
}

// ============================================================================
// MLP (Multi-Layer Perceptron)
// ============================================================================

/// Simple MLP with configurable layers
#[derive(Debug)]
pub struct MLP {
    layers: Vec<Linear>,
    num_layers: usize,
    span: tracing::Span,
}

impl MLP {
    pub fn load(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let mut dims: Vec<usize> = vec![input_dim];
        dims.extend(vec![hidden_dim; num_layers - 1]);
        dims.push(output_dim);

        for i in 0..num_layers {
            let layer = candle_nn::linear(dims[i], dims[i + 1], vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            num_layers,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut out = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            if i < self.num_layers - 1 {
                out = out.relu()?;
            }
        }
        Ok(out)
    }
}

// ============================================================================
// Layer Normalization
// ============================================================================

#[derive(Debug)]
pub struct LayerNormWrapper {
    inner: LayerNorm,
    span: tracing::Span,
}

impl LayerNormWrapper {
    pub fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let inner = candle_nn::layer_norm(dim, eps, vb)?;
        Ok(Self {
            inner,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for LayerNormWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ============================================================================
// Patch Embedding (for DINOv2 backbone)
// ============================================================================

#[derive(Debug)]
pub struct PatchEmbedding {
    proj: Conv2d,
    #[allow(dead_code)]
    patch_size: usize,
    span: tracing::Span,
}

impl PatchEmbedding {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: patch_size,
            padding: 0,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_channels, embed_dim, patch_size, cfg, vb.pp("projection"))?;
        Ok(Self {
            proj,
            patch_size,
            span: tracing::span!(tracing::Level::TRACE, "patch-embed"),
        })
    }
}

impl Module for PatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Input: (B, C, H, W)
        // Output: (B, num_patches, embed_dim)
        let x = self.proj.forward(xs)?;
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, h * w))?.permute((0, 2, 1))
    }
}

// ============================================================================
// Multi-Head Self-Attention
// ============================================================================

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    span: tracing::Span,
}

impl MultiHeadSelfAttention {
    pub fn load(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let q_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
            span: tracing::span!(tracing::Level::TRACE, "mhsa"),
        })
    }

    pub fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, n, c) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (B, num_heads, N, head_dim)
        let q = q
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        // Attention scores
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = match attn_mask {
            Some(mask) => (attn + mask)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Apply attention to values
        let out = attn.matmul(&v)?;
        let out = out.permute((0, 2, 1, 3))?.reshape((b, n, c))?;
        self.out_proj.forward(&out)
    }
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

#[derive(Debug)]
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    span: tracing::Span,
}

impl FeedForward {
    pub fn load(vb: VarBuilder, dim: usize, hidden_dim: usize) -> Result<Self> {
        let fc1 = candle_nn::linear(dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, dim, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "ffn"),
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = self.fc1.forward(xs)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ============================================================================
// Transformer Encoder Block (for DINOv2)
// ============================================================================

#[derive(Debug)]
pub struct TransformerEncoderBlock {
    norm1: LayerNormWrapper,
    attn: MultiHeadSelfAttention,
    norm2: LayerNormWrapper,
    mlp: FeedForward,
    span: tracing::Span,
}

impl TransformerEncoderBlock {
    pub fn load(
        vb: VarBuilder,
        embed_dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
    ) -> Result<Self> {
        let mlp_hidden_dim = (embed_dim as f64 * mlp_ratio) as usize;

        let norm1 = LayerNormWrapper::load(vb.pp("norm1"), embed_dim, 1e-6)?;
        let attn = MultiHeadSelfAttention::load(vb.pp("attention"), embed_dim, num_heads)?;
        let norm2 = LayerNormWrapper::load(vb.pp("norm2"), embed_dim, 1e-6)?;
        let mlp = FeedForward::load(vb.pp("mlp"), embed_dim, mlp_hidden_dim)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            span: tracing::span!(tracing::Level::TRACE, "encoder-block"),
        })
    }
}

impl Module for TransformerEncoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Pre-norm architecture
        let residual = xs;
        let x = self.norm1.forward(xs)?;
        let x = self.attn.forward(&x, None)?;
        let x = (residual + &x)?;

        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

// ============================================================================
// DINOv2 Backbone (Simplified)
// ============================================================================

#[derive(Debug)]
pub struct DINOv2Backbone {
    patch_embed: PatchEmbedding,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<TransformerEncoderBlock>,
    norm: LayerNormWrapper,
    out_feature_indexes: Vec<usize>,
    embed_dim: usize,
    span: tracing::Span,
}

impl DINOv2Backbone {
    pub fn load(vb: VarBuilder, config: &RFDETRConfig) -> Result<Self> {
        let embed_dim = config.encoder_embed_dim;
        let num_heads = if embed_dim == 384 { 6 } else { 12 }; // small: 6, base/large: 12
        let mlp_ratio = 4.0;

        let patch_embed = PatchEmbedding::load(
            vb.pp("embeddings.patch_embeddings"),
            3,
            embed_dim,
            config.patch_size,
        )?;

        let cls_token = vb.get((1, 1, embed_dim), "embeddings.cls_token")?;

        // Calculate expected number of position embeddings
        let num_patches = (config.positional_encoding_size * config.positional_encoding_size) + 1;
        let pos_embed = vb.get(
            (1, num_patches, embed_dim),
            "embeddings.position_embeddings",
        )?;

        let num_blocks = *config.out_feature_indexes.last().unwrap_or(&12) + 1;
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block = TransformerEncoderBlock::load(
                vb.pp(format!("encoder.layer.{}", i)),
                embed_dim,
                num_heads,
                mlp_ratio,
            )?;
            blocks.push(block);
        }

        let norm = LayerNormWrapper::load(vb.pp("layernorm"), embed_dim, 1e-6)?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            out_feature_indexes: config.out_feature_indexes.clone(),
            embed_dim,
            span: tracing::span!(tracing::Level::TRACE, "dinov2"),
        })
    }

    /// Forward pass returning features at specified layer indexes
    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let (b, _c, _h, _w) = xs.dims4()?;

        // Patch embedding
        let x = self.patch_embed.forward(xs)?;
        let _n = x.dim(1)?;

        // Add CLS token
        let cls_tokens = self.cls_token.broadcast_as((b, 1, self.embed_dim))?;
        let x = Tensor::cat(&[&cls_tokens, &x], 1)?;

        // Add position embeddings (interpolate if needed)
        let x = (&x + &self.pos_embed.i((.., ..x.dim(1)?, ..))?)?;

        // Collect features at specified layers
        let mut features = Vec::new();
        let mut current = x;

        for (i, block) in self.blocks.iter().enumerate() {
            current = block.forward(&current)?;
            if self.out_feature_indexes.contains(&i) {
                // Remove CLS token and reshape to spatial format
                let feat = current.i((.., 1.., ..))?;
                features.push(feat);
            }
        }

        // Apply final norm to last feature
        if let Some(last) = features.last_mut() {
            *last = self.norm.forward(last)?;
        }

        Ok(features)
    }
}

// ============================================================================
// Multi-Scale Projector
// ============================================================================

#[derive(Debug)]
pub struct MultiScaleProjector {
    input_proj: Vec<Linear>,
    #[allow(dead_code)]
    scale_factors: Vec<f64>,
    span: tracing::Span,
}

impl MultiScaleProjector {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        scale_factors: Vec<f64>,
    ) -> Result<Self> {
        let mut input_proj = Vec::with_capacity(scale_factors.len());
        for i in 0..scale_factors.len() {
            let proj = candle_nn::linear(
                in_channels,
                out_channels,
                vb.pp(format!("input_proj.{}", i)),
            )?;
            input_proj.push(proj);
        }

        Ok(Self {
            input_proj,
            scale_factors,
            span: tracing::span!(tracing::Level::TRACE, "projector"),
        })
    }

    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let mut outputs = Vec::with_capacity(self.input_proj.len());

        // Use the last feature from backbone
        let feat = features
            .last()
            .ok_or_else(|| candle::Error::Msg("No features provided".to_string()))?;

        for (_i, proj) in self.input_proj.iter().enumerate() {
            let projected = proj.forward(feat)?;
            // Reshape to spatial format if needed
            outputs.push(projected);
        }

        Ok(outputs)
    }
}

// ============================================================================
// Sine Position Embedding
// ============================================================================

#[allow(dead_code)]
pub fn generate_sine_position_embedding(
    h: usize,
    w: usize,
    hidden_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let num_pos_feats = hidden_dim / 2;
    let temperature = 10000.0_f64;
    let scale = 2.0 * std::f64::consts::PI;

    // Create position indices
    let y_positions: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |_| (y as f32 + 0.5) / h as f32 * scale as f32))
        .collect();
    let x_positions: Vec<f32> = (0..h)
        .flat_map(|_| (0..w).map(|x| (x as f32 + 0.5) / w as f32 * scale as f32))
        .collect();

    let y_embed = Tensor::from_vec(y_positions, (1, h, w), device)?.to_dtype(dtype)?;
    let x_embed = Tensor::from_vec(x_positions, (1, h, w), device)?.to_dtype(dtype)?;

    // Create dimension indices
    let dim_t: Vec<f32> = (0..num_pos_feats)
        .map(|i| {
            let exp = 2.0 * (i as f64 / 2.0).floor() / num_pos_feats as f64;
            (temperature.powf(exp)) as f32
        })
        .collect();
    let dim_t = Tensor::from_vec(dim_t, (1, 1, 1, num_pos_feats), device)?.to_dtype(dtype)?;

    // Compute sin/cos embeddings
    let pos_x = x_embed.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;
    let pos_y = y_embed.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;

    let pos_x_sin = pos_x.sin()?;
    let pos_x_cos = pos_x.cos()?;
    let pos_y_sin = pos_y.sin()?;
    let pos_y_cos = pos_y.cos()?;

    // Interleave sin and cos
    let pos_x =
        Tensor::stack(&[&pos_x_sin, &pos_x_cos], D::Minus1)?.reshape((1, h, w, hidden_dim / 2))?;
    let pos_y =
        Tensor::stack(&[&pos_y_sin, &pos_y_cos], D::Minus1)?.reshape((1, h, w, hidden_dim / 2))?;

    Tensor::cat(&[&pos_y, &pos_x], D::Minus1)?.permute((0, 3, 1, 2))
}

// ============================================================================
// Transformer Decoder Layer
// ============================================================================

#[derive(Debug)]
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadSelfAttention,
    norm1: LayerNormWrapper,
    cross_attn_q_proj: Linear,
    cross_attn_kv_proj: Linear,
    cross_attn_out_proj: Linear,
    norm2: LayerNormWrapper,
    linear1: Linear,
    linear2: Linear,
    norm3: LayerNormWrapper,
    num_heads: usize,
    head_dim: usize,
    span: tracing::Span,
}

impl TransformerDecoderLayer {
    pub fn load(vb: VarBuilder, config: &RFDETRConfig) -> Result<Self> {
        let d_model = config.hidden_dim;
        let sa_nhead = config.sa_nheads;
        let ca_nhead = config.ca_nheads;
        let dim_feedforward = config.dim_feedforward;

        let self_attn = MultiHeadSelfAttention::load(vb.pp("self_attn"), d_model, sa_nhead)?;
        let norm1 = LayerNormWrapper::load(vb.pp("norm1"), d_model, 1e-5)?;

        // Cross attention - simplified version without deformable attention
        let cross_attn_q_proj = candle_nn::linear(d_model, d_model, vb.pp("cross_attn.q_proj"))?;
        let cross_attn_kv_proj =
            candle_nn::linear(d_model, d_model * 2, vb.pp("cross_attn.kv_proj"))?;
        let cross_attn_out_proj =
            candle_nn::linear(d_model, d_model, vb.pp("cross_attn.out_proj"))?;
        let norm2 = LayerNormWrapper::load(vb.pp("norm2"), d_model, 1e-5)?;

        let linear1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("linear2"))?;
        let norm3 = LayerNormWrapper::load(vb.pp("norm3"), d_model, 1e-5)?;

        Ok(Self {
            self_attn,
            norm1,
            cross_attn_q_proj,
            cross_attn_kv_proj,
            cross_attn_out_proj,
            norm2,
            linear1,
            linear2,
            norm3,
            num_heads: ca_nhead,
            head_dim: d_model / ca_nhead,
            span: tracing::span!(tracing::Level::TRACE, "decoder-layer"),
        })
    }

    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: &Tensor,
        _pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, n, c) = tgt.dims3()?;

        // Self attention
        let q = (tgt + query_pos)?;
        let tgt2 = self.self_attn.forward(&q, None)?;
        let tgt = (tgt + &tgt2)?;
        let tgt = self.norm1.forward(&tgt)?;

        // Cross attention (simplified - standard attention instead of deformable)
        let q = self.cross_attn_q_proj.forward(&(&tgt + query_pos)?)?;
        let kv = self.cross_attn_kv_proj.forward(memory)?;
        let kv_chunks = kv.chunk(2, D::Minus1)?;
        let k = &kv_chunks[0];
        let v = &kv_chunks[1];

        // Reshape for multi-head attention
        let m = memory.dim(1)?;
        let q = q
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, m, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, m, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let tgt2 = attn.matmul(&v)?;
        let tgt2 = tgt2.permute((0, 2, 1, 3))?.reshape((b, n, c))?;
        let tgt2 = self.cross_attn_out_proj.forward(&tgt2)?;

        let tgt = (&tgt + &tgt2)?;
        let tgt = self.norm2.forward(&tgt)?;

        // FFN
        let tgt2 = self.linear1.forward(&tgt)?;
        let tgt2 = tgt2.relu()?;
        let tgt2 = self.linear2.forward(&tgt2)?;
        let tgt = (&tgt + &tgt2)?;
        self.norm3.forward(&tgt)
    }
}

// ============================================================================
// Transformer Decoder
// ============================================================================

#[derive(Debug)]
pub struct TransformerDecoder {
    layers: Vec<TransformerDecoderLayer>,
    norm: Option<LayerNormWrapper>,
    ref_point_head: MLP,
    #[allow(dead_code)]
    bbox_reparam: bool,
    span: tracing::Span,
}

impl TransformerDecoder {
    pub fn load(vb: VarBuilder, config: &RFDETRConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.dec_layers);
        for i in 0..config.dec_layers {
            let layer =
                TransformerDecoderLayer::load(vb.pp(format!("decoder.layers.{}", i)), config)?;
            layers.push(layer);
        }

        let norm = Some(LayerNormWrapper::load(
            vb.pp("decoder.norm"),
            config.hidden_dim,
            1e-5,
        )?);

        let ref_point_head = MLP::load(
            vb.pp("decoder.ref_point_head"),
            config.hidden_dim * 2,
            config.hidden_dim,
            config.hidden_dim,
            2,
        )?;

        Ok(Self {
            layers,
            norm,
            ref_point_head,
            bbox_reparam: config.bbox_reparam,
            span: tracing::span!(tracing::Level::TRACE, "decoder"),
        })
    }

    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        refpoints: &Tensor,
        pos: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        // Generate query position embeddings from reference points
        let query_pos = self.generate_query_pos(refpoints)?;

        let mut output = tgt.clone();
        for layer in &self.layers {
            output = layer.forward(&output, memory, &query_pos, pos)?;
        }

        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }

        Ok((output, refpoints.clone()))
    }

    fn generate_query_pos(&self, refpoints: &Tensor) -> Result<Tensor> {
        // Generate sinusoidal embeddings for reference points
        let sine_embed = generate_sine_embed_for_position(refpoints)?;
        self.ref_point_head.forward(&sine_embed)
    }
}

/// Generate sinusoidal embeddings for position tensor
fn generate_sine_embed_for_position(pos_tensor: &Tensor) -> Result<Tensor> {
    let scale = 2.0 * std::f64::consts::PI;
    let dim = 128_usize; // d_model / 2
    let device = pos_tensor.device();
    let dtype = pos_tensor.dtype();

    let dim_t: Vec<f64> = (0..dim)
        .map(|i| {
            let exp = 2.0 * (i as f64 / 2.0).floor() / dim as f64;
            10000_f64.powf(exp)
        })
        .collect();
    let dim_t = Tensor::from_vec(dim_t, (dim,), device)?.to_dtype(dtype)?;

    let (b, n, d) = pos_tensor.dims3()?;

    // Extract x, y coordinates
    let x = (pos_tensor.i((.., .., 0))? * scale)?;
    let y = (pos_tensor.i((.., .., 1))? * scale)?;

    // Compute position encodings
    let pos_x = x.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;
    let pos_y = y.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;

    let pos_x_sin = pos_x.sin()?;
    let pos_x_cos = pos_x.cos()?;
    let pos_y_sin = pos_y.sin()?;
    let pos_y_cos = pos_y.cos()?;

    // Interleave and concatenate
    let pos_x = Tensor::stack(&[&pos_x_sin, &pos_x_cos], D::Minus1)?.reshape((b, n, dim))?;
    let pos_y = Tensor::stack(&[&pos_y_sin, &pos_y_cos], D::Minus1)?.reshape((b, n, dim))?;

    if d == 2 {
        Tensor::cat(&[&pos_y, &pos_x], D::Minus1)
    } else if d == 4 {
        // Also encode width and height
        let w = (pos_tensor.i((.., .., 2))? * scale)?;
        let h = (pos_tensor.i((.., .., 3))? * scale)?;

        let pos_w = w.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;
        let pos_h = h.unsqueeze(D::Minus1)?.broadcast_div(&dim_t)?;

        let pos_w =
            Tensor::stack(&[&pos_w.sin()?, &pos_w.cos()?], D::Minus1)?.reshape((b, n, dim))?;
        let pos_h =
            Tensor::stack(&[&pos_h.sin()?, &pos_h.cos()?], D::Minus1)?.reshape((b, n, dim))?;

        Tensor::cat(&[&pos_y, &pos_x, &pos_w, &pos_h], D::Minus1)
    } else {
        candle::bail!("Unsupported position tensor dimension: {}", d)
    }
}

// ============================================================================
// Detection Head
// ============================================================================

#[derive(Debug)]
pub struct DetectionHead {
    class_embed: Linear,
    bbox_embed: MLP,
    span: tracing::Span,
}

impl DetectionHead {
    pub fn load(vb: VarBuilder, config: &RFDETRConfig) -> Result<Self> {
        let class_embed =
            candle_nn::linear(config.hidden_dim, config.num_classes, vb.pp("class_embed"))?;
        let bbox_embed = MLP::load(
            vb.pp("bbox_embed"),
            config.hidden_dim,
            config.hidden_dim,
            4,
            3,
        )?;

        Ok(Self {
            class_embed,
            bbox_embed,
            span: tracing::span!(tracing::Level::TRACE, "detection-head"),
        })
    }

    pub fn forward(
        &self,
        hs: &Tensor,
        ref_unsigmoid: &Tensor,
        bbox_reparam: bool,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        // Classification logits
        let outputs_class = self.class_embed.forward(hs)?;

        // Bounding box prediction
        let bbox_delta = self.bbox_embed.forward(hs)?;

        let outputs_coord = if bbox_reparam {
            // Reparameterized bbox prediction
            let cxcy_delta = bbox_delta.i((.., .., ..2))?;
            let wh_delta = bbox_delta.i((.., .., 2..))?;
            let ref_wh = ref_unsigmoid.i((.., .., 2..))?;
            let ref_cxcy = ref_unsigmoid.i((.., .., ..2))?;

            let cxcy = ((cxcy_delta * &ref_wh)? + ref_cxcy)?;
            let wh = (wh_delta.exp()? * ref_wh)?;
            Tensor::cat(&[&cxcy, &wh], D::Minus1)?
        } else {
            candle_nn::ops::sigmoid(&(bbox_delta + ref_unsigmoid)?)?
        };

        Ok((outputs_class, outputs_coord))
    }
}

// ============================================================================
// RF-DETR Model
// ============================================================================

#[derive(Debug)]
pub struct RFDETR {
    pub config: RFDETRConfig,
    backbone: DINOv2Backbone,
    projector: MultiScaleProjector,
    decoder: TransformerDecoder,
    detection_head: DetectionHead,
    refpoint_embed: Embedding,
    query_feat: Embedding,
    enc_output: Vec<Linear>,
    enc_output_norm: Vec<LayerNormWrapper>,
    enc_out_class_embed: Vec<Linear>,
    enc_out_bbox_embed: Vec<MLP>,
    span: tracing::Span,
}

impl RFDETR {
    pub fn load(vb: VarBuilder, config: RFDETRConfig) -> Result<Self> {
        let backbone = DINOv2Backbone::load(vb.pp("backbone.0.encoder"), &config)?;

        let scale_factors = config
            .projector_scales
            .iter()
            .map(|s| match s.as_str() {
                "P3" => 2.0,
                "P4" => 1.0,
                "P5" => 0.5,
                "P6" => 0.25,
                _ => 1.0,
            })
            .collect();

        let projector = MultiScaleProjector::load(
            vb.pp("backbone.0.projector"),
            config.encoder_embed_dim,
            config.hidden_dim,
            scale_factors,
        )?;

        let decoder = TransformerDecoder::load(vb.pp("transformer"), &config)?;
        let detection_head = DetectionHead::load(vb.clone(), &config)?;

        // Query embeddings
        let refpoint_embed = candle_nn::embedding(config.num_queries, 4, vb.pp("refpoint_embed"))?;
        let query_feat =
            candle_nn::embedding(config.num_queries, config.hidden_dim, vb.pp("query_feat"))?;

        // Two-stage encoder outputs (only 1 group for inference)
        let mut enc_output = Vec::new();
        let mut enc_output_norm = Vec::new();
        let mut enc_out_class_embed = Vec::new();
        let mut enc_out_bbox_embed = Vec::new();

        if config.two_stage {
            enc_output.push(candle_nn::linear(
                config.hidden_dim,
                config.hidden_dim,
                vb.pp("transformer.enc_output.0"),
            )?);
            enc_output_norm.push(LayerNormWrapper::load(
                vb.pp("transformer.enc_output_norm.0"),
                config.hidden_dim,
                1e-5,
            )?);
            enc_out_class_embed.push(candle_nn::linear(
                config.hidden_dim,
                config.num_classes,
                vb.pp("transformer.enc_out_class_embed.0"),
            )?);
            enc_out_bbox_embed.push(MLP::load(
                vb.pp("transformer.enc_out_bbox_embed.0"),
                config.hidden_dim,
                config.hidden_dim,
                4,
                3,
            )?);
        }

        Ok(Self {
            config,
            backbone,
            projector,
            decoder,
            detection_head,
            refpoint_embed,
            query_feat,
            enc_output,
            enc_output_norm,
            enc_out_class_embed,
            enc_out_bbox_embed,
            span: tracing::span!(tracing::Level::TRACE, "rf-detr"),
        })
    }

    /// Forward pass for inference
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let (b, _c, _h, _w) = x.dims4()?;

        // Backbone forward
        let features = self.backbone.forward(x)?;

        // Project features
        let projected = self.projector.forward(&features)?;

        // Flatten and concatenate multi-scale features
        let mut src_flatten = Vec::new();
        let mut spatial_shapes = Vec::new();

        for feat in &projected {
            let (_b, n, _c) = feat.dims3()?;
            // Approximate spatial shape from sequence length
            let side = (n as f64).sqrt() as usize;
            spatial_shapes.push((side, side));
            src_flatten.push(feat.clone());
        }

        let memory = if src_flatten.len() == 1 {
            src_flatten[0].clone()
        } else {
            Tensor::cat(&src_flatten.iter().collect::<Vec<_>>(), 1)?
        };

        // Two-stage proposal generation
        let (tgt, refpoints) = if self.config.two_stage && !self.enc_output.is_empty() {
            self.generate_two_stage_proposals(&memory, &spatial_shapes, b)?
        } else {
            // Use learned queries
            let query_feat = self.query_feat.embeddings();
            let refpoint = self.refpoint_embed.embeddings();
            (
                query_feat.unsqueeze(0)?.broadcast_as((
                    b,
                    self.config.num_queries,
                    self.config.hidden_dim,
                ))?,
                refpoint
                    .unsqueeze(0)?
                    .broadcast_as((b, self.config.num_queries, 4))?,
            )
        };

        // Transformer decoder
        let (hs, ref_unsigmoid) = self.decoder.forward(&tgt, &memory, &refpoints, None)?;

        // Detection head
        self.detection_head
            .forward(&hs, &ref_unsigmoid, self.config.bbox_reparam)
    }

    fn generate_two_stage_proposals(
        &self,
        memory: &Tensor,
        spatial_shapes: &[(usize, usize)],
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _device = memory.device();
        let _dtype = memory.dtype();

        // Generate encoder output proposals
        let (output_memory, output_proposals) =
            self.gen_encoder_output_proposals(memory, spatial_shapes, batch_size)?;

        // Apply encoder output projection
        let output_memory = self.enc_output[0].forward(&output_memory)?;
        let output_memory = self.enc_output_norm[0].forward(&output_memory)?;

        // Get classification scores
        let enc_outputs_class = self.enc_out_class_embed[0].forward(&output_memory)?;

        // Get top-k proposals
        let topk = self.config.num_queries.min(enc_outputs_class.dim(1)?);
        let max_scores = enc_outputs_class.max(D::Minus1)?;
        let topk_indices = max_scores.arg_sort_last_dim(false)?.i((.., ..topk))?;

        // Gather top-k proposals and memory
        let refpoints = self.gather_topk(&output_proposals, &topk_indices)?;

        // Compute bounding box deltas
        let bbox_delta = self.enc_out_bbox_embed[0].forward(&output_memory)?;
        let bbox_delta = self.gather_topk(&bbox_delta, &topk_indices)?;

        // Reparameterized bbox prediction
        let refpoints = if self.config.bbox_reparam {
            let cxcy_delta = bbox_delta.i((.., .., ..2))?;
            let wh_delta = bbox_delta.i((.., .., 2..))?;
            let ref_wh = refpoints.i((.., .., 2..))?;
            let ref_cxcy = refpoints.i((.., .., ..2))?;

            let cxcy = ((cxcy_delta * &ref_wh)? + ref_cxcy)?;
            let wh = (wh_delta.exp()? * ref_wh)?;
            Tensor::cat(&[&cxcy, &wh], D::Minus1)?
        } else {
            candle_nn::ops::sigmoid(&(bbox_delta + &refpoints)?)?
        };

        let tgt = self.gather_topk(&output_memory, &topk_indices)?;

        Ok((tgt, refpoints))
    }

    fn gen_encoder_output_proposals(
        &self,
        memory: &Tensor,
        spatial_shapes: &[(usize, usize)],
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = memory.device();
        let dtype = memory.dtype();

        let mut proposals = Vec::new();
        let mut _cur = 0;

        for (lvl, &(h, w)) in spatial_shapes.iter().enumerate() {
            // Generate grid
            let mut grid_data = Vec::with_capacity(h * w * 4);
            for y in 0..h {
                for x in 0..w {
                    let cx = (x as f32 + 0.5) / w as f32;
                    let cy = (y as f32 + 0.5) / h as f32;
                    let wh = 0.05 * (2.0_f32).powi(lvl as i32);
                    grid_data.extend_from_slice(&[cx, cy, wh, wh]);
                }
            }
            let grid = Tensor::from_vec(grid_data, (1, h * w, 4), device)?
                .to_dtype(dtype)?
                .broadcast_as((batch_size, h * w, 4))?;
            proposals.push(grid);
            _cur += h * w;
        }

        let output_proposals = if proposals.len() == 1 {
            proposals.into_iter().next().unwrap()
        } else {
            Tensor::cat(&proposals.iter().collect::<Vec<_>>(), 1)?
        };

        // Unsigmoid (inverse sigmoid) for reparameterization
        let eps = 1e-5;
        let output_proposals = if self.config.bbox_reparam {
            output_proposals.clone()
        } else {
            let clamped = output_proposals.clamp(eps, 1.0 - eps)?;
            (clamped.clone() / (1.0 - &clamped)?)?.log()?
        };

        Ok((memory.clone(), output_proposals))
    }

    fn gather_topk(&self, tensor: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (b, _n, c) = tensor.dims3()?;
        let k = indices.dim(1)?;

        // Expand indices for gather
        let indices_expanded = indices.unsqueeze(D::Minus1)?.broadcast_as((b, k, c))?;
        tensor.gather(&indices_expanded, 1)
    }
}

// ============================================================================
// Post-Processing
// ============================================================================

/// Detection result for a single image
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub score: f32,
    pub class_id: usize,
}

/// Post-process model outputs to get final detections
pub fn postprocess(
    pred_logits: &Tensor,
    pred_boxes: &Tensor,
    target_size: (usize, usize),
    num_select: usize,
    threshold: f32,
) -> Result<Vec<Detection>> {
    let pred_logits = pred_logits.to_device(&Device::Cpu)?;
    let pred_boxes = pred_boxes.to_device(&Device::Cpu)?;

    let (b, _n, num_classes) = pred_logits.dims3()?;
    assert_eq!(b, 1, "Batch size must be 1 for post-processing");

    // Sigmoid to get probabilities
    let prob = candle_nn::ops::sigmoid(&pred_logits)?;
    let prob = prob.squeeze(0)?; // (N, num_classes)
    let boxes = pred_boxes.squeeze(0)?; // (N, 4)

    // Get top-k predictions across all classes
    let prob_flat = prob.flatten_all()?;
    let topk = num_select.min(prob_flat.dim(0)?);

    let topk_indices = prob_flat.arg_sort_last_dim(false)?.i(..topk)?;
    let topk_indices_vec: Vec<u32> = topk_indices.to_vec1()?;

    let (h, w) = target_size;
    let mut detections = Vec::new();

    for &idx in &topk_indices_vec {
        let idx = idx as usize;
        let query_idx = idx / num_classes;
        let class_id = idx % num_classes;

        let score: f32 = prob.i((query_idx, class_id))?.to_scalar()?;
        if score < threshold {
            continue;
        }

        // Get bbox (cx, cy, w, h) format
        let bbox_vec: Vec<f32> = boxes.i(query_idx)?.to_vec1()?;
        let cx = bbox_vec[0];
        let cy = bbox_vec[1];
        let bw = bbox_vec[2];
        let bh = bbox_vec[3];

        // Convert to (x1, y1, x2, y2) format and scale to image size
        let x1 = ((cx - bw / 2.0) * w as f32).max(0.0);
        let y1 = ((cy - bh / 2.0) * h as f32).max(0.0);
        let x2 = ((cx + bw / 2.0) * w as f32).min(w as f32);
        let y2 = ((cy + bh / 2.0) * h as f32).min(h as f32);

        detections.push(Detection {
            bbox: [x1, y1, x2, y2],
            score,
            class_id,
        });
    }

    Ok(detections)
}

/// Apply non-maximum suppression to detections
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    // Sort by score descending
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            if detections[i].class_id != detections[j].class_id {
                continue;
            }
            let iou = compute_iou(&detections[i].bbox, &detections[j].bbox);
            if iou > iou_threshold {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

fn compute_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union_area = area1 + area2 - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}
