//! RF-DETR Model Configuration
//!
//! This module defines the configuration structures for RF-DETR model variants.

/// Encoder type for the backbone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    /// DINOv2 with windowed attention, small variant (384 hidden dim)
    Dinov2WindowedSmall,
    /// DINOv2 with windowed attention, base variant (768 hidden dim)
    Dinov2WindowedBase,
}

impl EncoderType {
    /// Returns the hidden size of the encoder
    pub fn hidden_size(&self) -> usize {
        match self {
            EncoderType::Dinov2WindowedSmall => 384,
            EncoderType::Dinov2WindowedBase => 768,
        }
    }

    /// Returns the number of attention heads in the encoder
    pub fn num_attention_heads(&self) -> usize {
        match self {
            EncoderType::Dinov2WindowedSmall => 6,
            EncoderType::Dinov2WindowedBase => 12,
        }
    }
}

/// Projector scale levels for multi-scale feature extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorScale {
    P3,
    P4,
    P5,
}

impl ProjectorScale {
    /// Returns the scale factor relative to P4 (which is 1.0)
    pub fn scale_factor(&self) -> f64 {
        match self {
            ProjectorScale::P3 => 2.0,
            ProjectorScale::P4 => 1.0,
            ProjectorScale::P5 => 0.5,
        }
    }
}

/// Configuration for RF-DETR models
#[derive(Debug, Clone)]
pub struct RfDetrConfig {
    // Backbone encoder configuration
    /// Type of encoder backbone
    pub encoder: EncoderType,
    /// Indices of layers to extract features from (0-indexed)
    pub out_feature_indexes: Vec<usize>,
    /// Patch size for the vision transformer
    pub patch_size: usize,
    /// Number of windows for windowed attention
    pub num_windows: usize,

    // Projector configuration
    /// Projector scale levels to use
    pub projector_scale: Vec<ProjectorScale>,

    // Transformer decoder configuration
    /// Number of decoder layers
    pub dec_layers: usize,
    /// Hidden dimension for the transformer
    pub hidden_dim: usize,
    /// Number of self-attention heads in decoder
    pub sa_nheads: usize,
    /// Number of cross-attention heads in decoder
    pub ca_nheads: usize,
    /// Number of sampling points for deformable attention
    pub dec_n_points: usize,

    // Query configuration
    /// Number of object queries
    pub num_queries: usize,
    /// Number of queries to select in two-stage
    pub num_select: usize,

    // Model behavior flags
    /// Whether to use two-stage detection
    pub two_stage: bool,
    /// Whether to use bbox reparameterization
    pub bbox_reparam: bool,
    /// Whether to use lite reference point refinement
    pub lite_refpoint_refine: bool,
    /// Whether to use layer normalization
    pub layer_norm: bool,

    // Input/output configuration
    /// Input image resolution (square)
    pub resolution: usize,
    /// Size of positional encoding grid
    pub positional_encoding_size: usize,
    /// Number of output classes (COCO = 91, including background)
    pub num_classes: usize,

    // Training-specific (kept for completeness)
    /// Number of groups for group DETR during training
    pub group_detr: usize,
}

impl Default for RfDetrConfig {
    fn default() -> Self {
        Self::small()
    }
}

impl RfDetrConfig {
    /// Create configuration for RF-DETR Nano model
    pub fn nano() -> Self {
        Self {
            encoder: EncoderType::Dinov2WindowedSmall,
            out_feature_indexes: vec![3, 6, 9, 12],
            patch_size: 16,
            num_windows: 2,
            projector_scale: vec![ProjectorScale::P4],
            dec_layers: 2,
            hidden_dim: 256,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            num_queries: 300,
            num_select: 300,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            resolution: 384,
            positional_encoding_size: 24,
            num_classes: 91,
            group_detr: 13,
        }
    }

    /// Create configuration for RF-DETR Small model
    pub fn small() -> Self {
        Self {
            encoder: EncoderType::Dinov2WindowedSmall,
            out_feature_indexes: vec![3, 6, 9, 12],
            patch_size: 16,
            num_windows: 2,
            projector_scale: vec![ProjectorScale::P4],
            dec_layers: 3,
            hidden_dim: 256,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            num_queries: 300,
            num_select: 300,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            resolution: 512,
            positional_encoding_size: 32,
            num_classes: 91,
            group_detr: 13,
        }
    }

    /// Create configuration for RF-DETR Medium model
    pub fn medium() -> Self {
        Self {
            encoder: EncoderType::Dinov2WindowedSmall,
            out_feature_indexes: vec![3, 6, 9, 12],
            patch_size: 16,
            num_windows: 2,
            projector_scale: vec![ProjectorScale::P4],
            dec_layers: 4,
            hidden_dim: 256,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            num_queries: 300,
            num_select: 300,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            resolution: 576,
            positional_encoding_size: 36,
            num_classes: 91,
            group_detr: 13,
        }
    }

    /// Create configuration for RF-DETR Base model
    pub fn base() -> Self {
        Self {
            encoder: EncoderType::Dinov2WindowedSmall,
            out_feature_indexes: vec![2, 5, 8, 11],
            patch_size: 14,
            num_windows: 4,
            projector_scale: vec![ProjectorScale::P4],
            dec_layers: 3,
            hidden_dim: 256,
            sa_nheads: 8,
            ca_nheads: 16,
            dec_n_points: 2,
            num_queries: 300,
            num_select: 300,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            resolution: 560,
            positional_encoding_size: 37,
            num_classes: 91,
            group_detr: 13,
        }
    }

    /// Create configuration for RF-DETR Large model
    pub fn large() -> Self {
        Self {
            encoder: EncoderType::Dinov2WindowedBase,
            out_feature_indexes: vec![2, 5, 8, 11],
            patch_size: 14,
            num_windows: 4,
            projector_scale: vec![ProjectorScale::P3, ProjectorScale::P5],
            dec_layers: 3,
            hidden_dim: 384,
            sa_nheads: 12,
            ca_nheads: 24,
            dec_n_points: 4,
            num_queries: 300,
            num_select: 300,
            two_stage: true,
            bbox_reparam: true,
            lite_refpoint_refine: true,
            layer_norm: true,
            resolution: 560,
            positional_encoding_size: 37,
            num_classes: 91,
            group_detr: 13,
        }
    }

    /// Get the number of feature levels from the projector scales
    pub fn num_feature_levels(&self) -> usize {
        self.projector_scale.len()
    }

    /// Get the number of encoder layers (from DINOv2, typically 12)
    pub fn num_encoder_layers(&self) -> usize {
        // DINOv2 small/base both have 12 layers
        12
    }

    /// Calculate the feature map size at P4 scale for the given resolution
    pub fn feature_map_size(&self) -> usize {
        self.resolution / self.patch_size
    }
}
