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
    /// These represent initial reference points (cx, cy, w, h)
    pub refpoint_embed: Tensor,
    /// Query feature embeddings: [num_queries, hidden_dim]
    /// These represent initial query features for the decoder
    pub query_feat: Tensor,
}

impl QueryEmbeddings {
    /// Load query embeddings from weights
    ///
    /// # Arguments
    /// * `vb` - VarBuilder containing the model weights
    /// * `num_queries` - Number of queries to use during inference (typically 300)
    /// * `hidden_dim` - Hidden dimension for query features (typically 256)
    /// * `group_detr` - Number of groups used during training (typically 13)
    ///
    /// # Returns
    /// QueryEmbeddings with weights loaded from the VarBuilder
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;

    /// Integration test comparing query embeddings against Python reference
    ///
    /// Run with: cargo test test_query_embeddings_against_python -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_query_embeddings_against_python() {
        const WEIGHTS_PATH: &str = "../../py/rfdetr/export/rfdetr-small.safetensors";
        const DEBUG_DIR: &str = "../../py/rfdetr/output";

        // Helper to load numpy array
        fn load_npy(path: &str) -> Tensor {
            use ndarray_npy::ReadNpyExt;
            let file = std::fs::File::open(path).expect(&format!("Failed to open {}", path));
            let arr: ndarray::ArrayD<f32> =
                ndarray::ArrayD::<f32>::read_npy(file).expect("Failed to parse npy");
            let shape: Vec<usize> = arr.shape().to_vec();
            let data: Vec<f32> = arr.into_iter().collect();
            Tensor::from_vec(data, shape, &Device::Cpu).expect("Failed to create tensor")
        }

        // Helper to compare tensors
        fn compare_tensors(name: &str, rust: &Tensor, python: &Tensor, max_diff_threshold: f32) {
            let rust = rust
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let python = python
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

            assert_eq!(
                rust.dims(),
                python.dims(),
                "{}: Shape mismatch: {:?} vs {:?}",
                name,
                rust.dims(),
                python.dims()
            );

            let diff = (&rust - &python).unwrap().abs().unwrap();
            let max_diff = diff
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mean_diff = diff.mean_all().unwrap().to_scalar::<f32>().unwrap();

            let rust_mean = rust.mean_all().unwrap().to_scalar::<f32>().unwrap();
            let python_mean = python.mean_all().unwrap().to_scalar::<f32>().unwrap();

            println!(
                "{}: max_diff={:.6}, mean_diff={:.6}",
                name, max_diff, mean_diff
            );
            println!(
                "  Rust mean: {:.6}, Python mean: {:.6}",
                rust_mean, python_mean
            );

            assert!(
                max_diff < max_diff_threshold,
                "{}: max_diff ({:.6}) exceeds threshold ({:.6})",
                name,
                max_diff,
                max_diff_threshold
            );
        }

        // Check files exist
        if !std::path::Path::new(WEIGHTS_PATH).exists() {
            println!("Skipping test: weights file not found at {}", WEIGHTS_PATH);
            return;
        }

        let device = Device::Cpu;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[WEIGHTS_PATH], DType::F32, &device)
                .expect("Failed to load weights")
        };

        // Load query embeddings
        // RF-DETR small: num_queries=300, hidden_dim=256, group_detr=13
        let query_embeddings =
            QueryEmbeddings::load(vb, 300, 256, 13).expect("Failed to load query embeddings");

        println!("Query embeddings loaded:");
        println!(
            "  refpoint_embed shape: {:?}",
            query_embeddings.refpoint_embed.dims()
        );
        println!(
            "  query_feat shape: {:?}",
            query_embeddings.query_feat.dims()
        );

        // Compare refpoint_embed (step 07)
        let ref_path = format!("{}/07_refpoint_embed.npy", DEBUG_DIR);
        if std::path::Path::new(&ref_path).exists() {
            let reference = load_npy(&ref_path);
            println!("\nReference refpoint_embed shape: {:?}", reference.dims());

            compare_tensors(
                "07_refpoint_embed",
                &query_embeddings.refpoint_embed,
                &reference,
                1e-6, // Should match exactly - just loading weights
            );
        } else {
            println!("Reference file not found: {}", ref_path);
        }

        // Compare query_feat (step 08)
        let ref_path = format!("{}/08_query_feat.npy", DEBUG_DIR);
        if std::path::Path::new(&ref_path).exists() {
            let reference = load_npy(&ref_path);
            println!("\nReference query_feat shape: {:?}", reference.dims());

            compare_tensors(
                "08_query_feat",
                &query_embeddings.query_feat,
                &reference,
                1e-6, // Should match exactly - just loading weights
            );
        } else {
            println!("Reference file not found: {}", ref_path);
        }
    }
}
