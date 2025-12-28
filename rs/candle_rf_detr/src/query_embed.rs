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
pub struct QueryEmbeddings {
    /// Reference point embeddings: [num_queries, 4]
    /// These represent initial reference points (cx, cy, w, h)
    refpoint_embed: Tensor,

    /// Query feature embeddings: [num_queries, hidden_dim]
    /// These represent initial query features for the decoder
    query_feat: Tensor,

    /// Number of queries to use during inference
    num_queries: usize,
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
            num_queries,
        })
    }

    /// Get reference point embeddings
    ///
    /// # Returns
    /// Tensor of shape [num_queries, 4] containing (cx, cy, w, h) reference points
    pub fn refpoint_embed(&self) -> &Tensor {
        &self.refpoint_embed
    }

    /// Get query feature embeddings
    ///
    /// # Returns
    /// Tensor of shape [num_queries, hidden_dim] containing query features
    pub fn query_feat(&self) -> &Tensor {
        &self.query_feat
    }

    /// Get number of queries
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Get reference point embeddings expanded for a batch
    ///
    /// # Arguments
    /// * `batch_size` - Batch size to expand to
    ///
    /// # Returns
    /// Tensor of shape [batch_size, num_queries, 4]
    pub fn refpoint_embed_batched(&self, batch_size: usize) -> Result<Tensor> {
        // [num_queries, 4] -> [1, num_queries, 4] -> [batch_size, num_queries, 4]
        let expanded = self.refpoint_embed.unsqueeze(0)?;
        expanded.repeat((batch_size, 1, 1))
    }

    /// Get query feature embeddings expanded for a batch
    ///
    /// # Arguments
    /// * `batch_size` - Batch size to expand to
    ///
    /// # Returns
    /// Tensor of shape [batch_size, num_queries, hidden_dim]
    pub fn query_feat_batched(&self, batch_size: usize) -> Result<Tensor> {
        // [num_queries, hidden_dim] -> [1, num_queries, hidden_dim] -> [batch_size, num_queries, hidden_dim]
        let expanded = self.query_feat.unsqueeze(0)?;
        expanded.repeat((batch_size, 1, 1))
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
            query_embeddings.refpoint_embed().dims()
        );
        println!(
            "  query_feat shape: {:?}",
            query_embeddings.query_feat().dims()
        );

        // Compare refpoint_embed (step 07)
        let ref_path = format!("{}/07_refpoint_embed.npy", DEBUG_DIR);
        if std::path::Path::new(&ref_path).exists() {
            let reference = load_npy(&ref_path);
            println!("\nReference refpoint_embed shape: {:?}", reference.dims());

            compare_tensors(
                "07_refpoint_embed",
                query_embeddings.refpoint_embed(),
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
                query_embeddings.query_feat(),
                &reference,
                1e-6, // Should match exactly - just loading weights
            );
        } else {
            println!("Reference file not found: {}", ref_path);
        }
    }

    #[test]
    fn test_query_embeddings_batched() {
        // Create dummy tensors for testing
        let device = Device::Cpu;
        let refpoint = Tensor::zeros((300, 4), DType::F32, &device).unwrap();
        let query = Tensor::zeros((300, 256), DType::F32, &device).unwrap();

        let embeddings = QueryEmbeddings {
            refpoint_embed: refpoint,
            query_feat: query,
            num_queries: 300,
        };

        // Test batched expansion
        let batch_size = 2;
        let refpoint_batched = embeddings.refpoint_embed_batched(batch_size).unwrap();
        let query_batched = embeddings.query_feat_batched(batch_size).unwrap();

        assert_eq!(refpoint_batched.dims(), &[2, 300, 4]);
        assert_eq!(query_batched.dims(), &[2, 300, 256]);
    }
}
