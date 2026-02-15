//! Query Embeddings for RF-DETR
//!
//! This module implements the query embeddings used in RF-DETR's transformer decoder.
//! There are two types of embeddings:
//!
//! 1. **Reference Point Embeddings (refpoint_embed)**: Shape [num_queries, 4]
//!    - Learned embeddings representing initial reference points for object detection
//!    - The 4 dimensions correspond to (cx, cy, w, h) in normalized coordinates
//!
//! 2. **Query Feature Embeddings (query_feat)**: Shape [num_queries, hidden_dim]
//!    - Learned embeddings representing initial query features for the transformer decoder
//!
//! During training, all `num_queries * group_detr` embeddings are used.
//! During inference, only the first `num_queries` embeddings are used.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Query embeddings for RF-DETR transformer decoder
#[derive(Debug)]
pub struct QueryEmbeddings {
    /// Reference point embeddings: [num_queries, 4]
    /// These represent initial reference points (cx, cy, w, h) for object detection.
    pub refpoint_embed: Tensor,
    /// Query feature embeddings: [num_queries, hidden_dim]
    /// These represent initial query features for the decoder.
    pub query_feat: Tensor,
}

impl QueryEmbeddings {
    /// # Arguments
    /// * `vb` - VarBuilder containing the model weights
    /// * `num_queries` - Number of queries to use during inference (typically 300)
    /// * `hidden_dim` - Hidden dimension for query features (typically 256)
    /// * `group_detr` - Number of groups used during training (typically 13)
    pub fn load(
        vb: VarBuilder,
        num_queries: usize,
        hidden_dim: usize,
        group_detr: usize,
    ) -> Result<Self> {
        // Full embedding sizes include group_detr multiplier
        let full_num_queries = num_queries * group_detr;

        // Load refpoint_embed.weight: [num_queries * group_detr, 4]
        let refpoint_embed_full = vb.get((full_num_queries, 4), "refpoint_embed.weight")?;

        // Load query_feat.weight: [num_queries * group_detr, hidden_dim]
        let query_feat_full = vb.get((full_num_queries, hidden_dim), "query_feat.weight")?;

        // For inference, we only use the first num_queries embeddings
        let refpoint_embed = refpoint_embed_full.narrow(0, 0, num_queries)?;
        let query_feat = query_feat_full.narrow(0, 0, num_queries)?;

        Ok(Self {
            refpoint_embed,
            query_feat,
        })
    }
}
