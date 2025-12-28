//! RF-DETR Model Implementation
//!
//! This module provides the main RF-DETR model structure and loading functionality.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::RfDetrConfig;
use crate::dino2::{DinoV2Encoder, Dinov2Config};
use crate::pos_enc::PositionEmbeddingSine;
use crate::projector::{MultiScaleProjector, ProjectorConfig};
use crate::query_embed::QueryEmbeddings;
use crate::transformer::{Mlp, Transformer};

/// RF-DETR Object Detection Model
///
/// This struct holds the loaded model weights and configuration
/// for performing object detection inference.
pub struct RfDetr {
    /// Model configuration
    pub config: RfDetrConfig,

    /// Backbone encoder (DINOv2)
    backbone_encoder: DinoV2Encoder,

    /// Feature projector (multi-scale)
    projector: MultiScaleProjector,

    /// Position encoding generator
    position_encoding: PositionEmbeddingSine,

    /// Query embeddings (refpoint_embed and query_feat)
    query_embeddings: QueryEmbeddings,

    /// Transformer (decoder + two-stage)
    transformer: Transformer,

    /// Class embedding head
    class_embed: Linear,

    /// Bbox embedding head
    bbox_embed: Mlp,
}

impl RfDetr {
    /// Load an RF-DETR model from weights
    ///
    /// # Arguments
    /// * `vb` - VarBuilder containing the model weights
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// A loaded RF-DETR model ready for inference
    pub fn load(vb: VarBuilder, config: &RfDetrConfig) -> Result<Self> {
        // Create DINOv2 config from RF-DETR config
        let dino_config = match config.encoder {
            crate::config::EncoderType::Dinov2WindowedSmall => Dinov2Config::small_windowed(
                config.resolution,
                config.patch_size,
                config.num_windows,
                &config.out_feature_indexes,
            ),
            crate::config::EncoderType::Dinov2WindowedBase => Dinov2Config::base_windowed(
                config.resolution,
                config.patch_size,
                config.num_windows,
                &config.out_feature_indexes,
            ),
        };

        // Load backbone encoder
        // Weight path: backbone.0.encoder.encoder.*
        let backbone_encoder =
            DinoV2Encoder::load(vb.pp("backbone.0.encoder.encoder"), &dino_config)?;

        // Load projector
        // Weight path: backbone.0.projector.*
        let projector_config = ProjectorConfig::small(
            config.hidden_dim,
            dino_config.hidden_size,
            config.out_feature_indexes.len(),
        );
        let projector =
            MultiScaleProjector::load(vb.pp("backbone.0.projector"), &projector_config)?;

        // Create position encoding (no weights to load - computed from formula)
        let position_encoding = PositionEmbeddingSine::for_rf_detr(config.hidden_dim);

        // Load query embeddings (refpoint_embed and query_feat)
        let query_embeddings = QueryEmbeddings::load(
            vb.clone(),
            config.num_queries,
            config.hidden_dim,
            config.group_detr,
        )?;

        // Load transformer
        // Weight path: transformer.*
        let dim_feedforward = 2048; // Standard for RF-DETR
        let transformer = Transformer::load(
            config.hidden_dim,
            config.sa_nheads,
            config.ca_nheads,
            config.num_queries,
            config.dec_layers,
            dim_feedforward,
            config.num_feature_levels(),
            config.dec_n_points,
            config.lite_refpoint_refine,
            config.bbox_reparam,
            config.num_classes,
            vb.pp("transformer"),
        )?;

        // Load class_embed
        // Weight path: class_embed.*
        let class_embed = linear(config.hidden_dim, config.num_classes, vb.pp("class_embed"))?;

        // Load bbox_embed
        // Weight path: bbox_embed.*
        let bbox_embed = Mlp::load(
            config.hidden_dim,
            config.hidden_dim,
            4,
            3,
            vb.pp("bbox_embed"),
        )?;

        Ok(Self {
            config: config.clone(),
            backbone_encoder,
            projector,
            position_encoding,
            query_embeddings,
            transformer,
            class_embed,
            bbox_embed,
        })
    }

    /// Run the backbone encoder to get multi-scale feature maps
    ///
    /// # Arguments
    /// * `pixel_values` - Input tensor of shape [batch_size, 3, height, width]
    ///
    /// # Returns
    /// A vector of feature maps at different scales, each with shape [batch_size, encoder_hidden_dim, h, w]
    pub fn backbone_encoder_forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        self.backbone_encoder.forward(pixel_values)
    }

    /// Run the projector to project encoder features to hidden_dim
    ///
    /// # Arguments
    /// * `encoder_outputs` - Vector of feature maps from backbone encoder
    ///
    /// # Returns
    /// A vector of projected feature maps, each with shape [batch_size, hidden_dim, h, w]
    pub fn projector_forward(&self, encoder_outputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.projector.forward(encoder_outputs)
    }

    /// Run the full backbone (encoder + projector)
    ///
    /// # Arguments
    /// * `pixel_values` - Input tensor of shape [batch_size, 3, height, width]
    ///
    /// # Returns
    /// A vector of projected feature maps, each with shape [batch_size, hidden_dim, h, w]
    pub fn backbone_forward(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let encoder_outputs = self.backbone_encoder_forward(pixel_values)?;
        self.projector_forward(&encoder_outputs)
    }

    /// Get query embeddings
    pub fn query_embeddings(&self) -> &QueryEmbeddings {
        &self.query_embeddings
    }

    /// Compute position encodings for feature maps
    ///
    /// # Arguments
    /// * `feature_maps` - Vector of feature maps from projector
    ///
    /// # Returns
    /// A vector of position encodings, each with shape [batch_size, hidden_dim, h, w]
    pub fn compute_position_encodings(
        &self,
        feature_maps: &[Tensor],
        device: &Device,
    ) -> Result<Vec<Tensor>> {
        let mut pos_encodings = Vec::with_capacity(feature_maps.len());
        for feat in feature_maps {
            let (batch_size, _channels, height, width) = feat.dims4()?;
            let pos = self
                .position_encoding
                .forward(batch_size, height, width, device)?;
            pos_encodings.push(pos);
        }
        Ok(pos_encodings)
    }

    /// Run transformer forward pass (steps 09-12)
    ///
    /// # Arguments
    /// * `projector_outputs` - Feature maps from projector
    /// * `position_encodings` - Position encodings
    ///
    /// # Returns
    /// (decoder_hs, decoder_ref, encoder_hs, encoder_ref)
    pub fn transformer_forward(
        &self,
        projector_outputs: &[Tensor],
        position_encodings: &[Tensor],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        self.transformer.forward(
            projector_outputs,
            position_encodings,
            self.query_embeddings.refpoint_embed(),
            self.query_embeddings.query_feat(),
        )
    }

    /// Run full inference on an input image
    ///
    /// # Arguments
    /// * `pixel_values` - Preprocessed input tensor [batch_size, 3, height, width]
    ///
    /// # Returns
    /// Tuple of (class_logits, bbox_predictions) where:
    /// - class_logits: [batch_size, num_queries, num_classes]
    /// - bbox_predictions: [batch_size, num_queries, 4] in (cx, cy, w, h) format
    pub fn forward(&self, pixel_values: &Tensor) -> Result<(Tensor, Tensor)> {
        // Steps 04-05: Backbone encoder + projector
        let projector_outputs = self.backbone_forward(pixel_values)?;

        // Step 06: Position encoding
        let position_encodings =
            self.compute_position_encodings(&projector_outputs, pixel_values.device())?;

        // Steps 09-12: Transformer
        let (decoder_hs, decoder_ref, _encoder_hs, _encoder_ref) =
            self.transformer_forward(&projector_outputs, &position_encodings)?;

        // Steps 13-17: Class and bbox predictions
        // Bbox prediction with reparameterization
        let outputs_coord = if self.config.bbox_reparam {
            let outputs_coord_delta = self.bbox_embed.forward(&decoder_hs)?;

            let ref_xy = decoder_ref.narrow(candle_core::D::Minus1, 0, 2)?;
            let ref_wh = decoder_ref.narrow(candle_core::D::Minus1, 2, 2)?;
            let delta_xy = outputs_coord_delta.narrow(candle_core::D::Minus1, 0, 2)?;
            let delta_wh = outputs_coord_delta.narrow(candle_core::D::Minus1, 2, 2)?;

            let outputs_coord_cxcy = delta_xy.mul(&ref_wh)?.add(&ref_xy)?;
            let outputs_coord_wh = delta_wh.exp()?.mul(&ref_wh)?;
            Tensor::cat(
                &[&outputs_coord_cxcy, &outputs_coord_wh],
                candle_core::D::Minus1,
            )?
        } else {
            let bbox_delta = self.bbox_embed.forward(&decoder_hs)?;
            candle_nn::ops::sigmoid(&(bbox_delta + &decoder_ref)?)?
        };

        // Classification
        let outputs_class = self.class_embed.forward(&decoder_hs)?;

        Ok((outputs_class, outputs_coord))
    }
}
