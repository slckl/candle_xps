pub mod dino2;
pub mod pos_enc;
pub mod projector;
pub mod query_embed;
pub mod segmentation_head;
pub mod transformer;

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::RfDetrConfig;
use crate::model::dino2::{Dinov2Backbone, Dinov2Config};
use crate::model::pos_enc::PositionEmbeddingSine;
use crate::model::projector::{MultiScaleProjector, ProjectorConfig};
use crate::model::query_embed::QueryEmbeddings;
use crate::model::segmentation_head::{SegmentationHead, SegmentationHeadConfig};
use crate::model::transformer::{Mlp, Transformer};

/// RF-DETR Object Detection/Instance Segmentation Model
///
/// This struct holds the loaded model weights and configuration
/// for performing object detection and optional instance segmentation inference.
pub struct RfDetr {
    /// Model configuration
    pub config: RfDetrConfig,
    /// Backbone encoder (DINOv2)
    backbone_encoder: Dinov2Backbone,
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
    /// Optional segmentation head
    segmentation_head: Option<SegmentationHead>,
}

impl RfDetr {
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
            Dinov2Backbone::load(vb.pp("backbone.0.encoder.encoder"), &dino_config)?;

        // TODO use values from config, instead of hardcoded...
        // Load projector
        // Weight path: backbone.0.projector.*
        // Use different config based on scale factors
        let projector_config = if config.projector_scale.len() == 1
            && config.projector_scale[0] == crate::config::ProjectorScale::P4
        {
            // Small/Medium/Nano: single scale at P4 (scale_factor=1.0)
            ProjectorConfig::small(
                config.hidden_dim,
                dino_config.hidden_size,
                config.out_feature_indexes.len(),
            )
        } else {
            // Large: P3+P5 (scale_factors=[2.0, 0.5])
            ProjectorConfig::large(
                config.hidden_dim,
                dino_config.hidden_size,
                config.out_feature_indexes.len(),
            )
        };
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
        // Dim is same for all RF-DETR pretrained models so far...
        let dim_feedforward = 2048;
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

        // Load segmentation head if enabled
        let segmentation_head = if config.segmentation_head {
            let seg_config = SegmentationHeadConfig::new(config.hidden_dim, config.seg_num_blocks);
            Some(SegmentationHead::load(
                vb.pp("segmentation_head"),
                &seg_config,
            )?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            backbone_encoder,
            projector,
            position_encoding,
            query_embeddings,
            transformer,
            class_embed,
            bbox_embed,
            segmentation_head,
        })
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

    /// Run full inference on an input image
    ///
    /// # Arguments
    /// * `pixel_values` - Preprocessed input tensor [batch_size, 3, height, width]
    ///
    /// # Returns
    /// Tuple of (class_logits, bbox_predictions, mask_logits) where:
    /// - class_logits: [batch_size, num_queries, num_classes]
    /// - bbox_predictions: [batch_size, num_queries, 4] in (cx, cy, w, h) format
    /// - mask_logits: optional, [batch_size, num_queries, H', W'] where H' = resolution / downsample_ratio
    pub fn forward(&self, pixel_values: &Tensor) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        // Dino2 windowed backbone -> multi-scale projector -> position encodings -> transformer -> class + bbox heads + seg head

        // Backbone -> projector
        // [batch_size, 3, height, width] -> [batch_size, encoder_hidden_dim, h, w]
        let encoder_outputs = self.backbone_encoder.forward(pixel_values)?;
        // [batch_size, encoder_hidden_dim, h, w] -> [batch_size, hidden_dim, h, w]
        let projector_outputs = self.projector.forward(&encoder_outputs)?;

        // Step 06: Position encoding
        let position_encodings =
            self.compute_position_encodings(&projector_outputs, pixel_values.device())?;

        // Steps 09-12: Transformer
        let (decoder_hs, decoder_ref, _encoder_hs, _encoder_ref) = self.transformer.forward(
            &projector_outputs,
            &position_encodings,
            &self.query_embeddings.refpoint_embed,
            &self.query_embeddings.query_feat,
        )?;

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

        // Optional segmentation masks.
        let mask_logits = if let Some(seg_head) = &self.segmentation_head {
            // Use the first projector output (srcs[0]) as spatial features
            let spatial_features = &projector_outputs[0];
            let (_, _, input_h, input_w) = pixel_values.dims4()?;
            let mask_logits = seg_head.forward(spatial_features, &decoder_hs, input_h, input_w)?;
            Some(mask_logits)
        } else {
            None
        };

        Ok((outputs_class, outputs_coord, mask_logits))
    }
}
