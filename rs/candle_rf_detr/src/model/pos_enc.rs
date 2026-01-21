//! Sinusoidal Position Encoding for RF-DETR
//!
//! This module implements the sinusoidal position encoding used in DETR-style models.
//! The position encoding is computed using sine and cosine functions at different
//! frequencies, similar to the original Transformer paper but generalized for 2D images.
//!
//! For RF-DETR:
//! - num_pos_feats = hidden_dim // 2 = 128 (for hidden_dim=256)
//! - temperature = 10000
//! - normalize = true
//! - scale = 2π
//!
//! Output shape: [batch_size, hidden_dim, height, width]

use candle_core::{Result, Tensor};

/// Configuration for sinusoidal position embedding
#[derive(Debug, Clone)]
pub struct PositionEmbeddingSineConfig {
    /// Number of position features (typically hidden_dim // 2)
    pub num_pos_feats: usize,
    /// Temperature for the frequency scaling (typically 10000)
    pub temperature: f64,
    /// Whether to normalize positions to [0, scale] range
    pub normalize: bool,
    /// Scale factor for normalization (typically 2π)
    pub scale: f64,
}

impl PositionEmbeddingSineConfig {
    /// Create default config for RF-DETR
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            num_pos_feats: hidden_dim / 2,
            temperature: 10000.0,
            normalize: true,
            scale: 2.0 * std::f64::consts::PI,
        }
    }
}

/// Sinusoidal Position Embedding
///
/// This is the standard position embedding used in DETR, generalized to work on images.
/// It creates position encodings using sine and cosine functions at different frequencies.
pub struct PositionEmbeddingSine {
    config: PositionEmbeddingSineConfig,
}

impl PositionEmbeddingSine {
    /// Create a new sinusoidal position embedding
    pub fn new(config: PositionEmbeddingSineConfig) -> Self {
        Self { config }
    }

    /// Create position embedding for RF-DETR with default config
    pub fn for_rf_detr(hidden_dim: usize) -> Self {
        Self::new(PositionEmbeddingSineConfig::new(hidden_dim))
    }

    /// Compute position encoding for a feature map
    ///
    /// # Arguments
    /// * `batch_size` - Batch size
    /// * `height` - Height of the feature map
    /// * `width` - Width of the feature map
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// Position encoding tensor of shape [batch_size, num_pos_feats * 2, height, width]
    pub fn forward(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        // Create mask (all False = all valid positions)
        // not_mask is all ones
        let not_mask = Tensor::ones((batch_size, height, width), candle_core::DType::F32, device)?;

        // Compute cumulative sum for y positions (along height dimension)
        // y_embed[b, i, j] = i + 1 (1-indexed row position, same for all columns)
        let y_embed = self.cumsum_2d(&not_mask, 1)?;

        // Compute cumulative sum for x positions (along width dimension)
        // x_embed[b, i, j] = j + 1 (1-indexed column position, same for all rows)
        let x_embed = self.cumsum_2d(&not_mask, 2)?;

        // Normalize if configured
        let (y_embed, x_embed) = if self.config.normalize {
            let eps = 1e-6;

            // Get the last value along each dimension for normalization
            // y_embed[:, -1:, :] gives the max y value (height)
            let y_max = y_embed.narrow(1, height - 1, 1)?;
            let y_embed = ((y_embed.broadcast_div(&(y_max + eps)?))? * self.config.scale)?;

            // x_embed[:, :, -1:] gives the max x value (width)
            let x_max = x_embed.narrow(2, width - 1, 1)?;
            let x_embed = ((x_embed.broadcast_div(&(x_max + eps)?))? * self.config.scale)?;

            (y_embed, x_embed)
        } else {
            (y_embed, x_embed)
        };

        // Create dimension tensor for frequency scaling
        // dim_t = temperature ** (2 * (i // 2) / num_pos_feats) for i in 0..num_pos_feats
        let dim_t = self.create_dim_tensor(device)?;

        // Expand embeddings for division: [B, H, W] -> [B, H, W, 1]
        let y_embed = y_embed.unsqueeze(3)?;
        let x_embed = x_embed.unsqueeze(3)?;

        // Divide by dimension tensor: [B, H, W, 1] / [num_pos_feats] -> [B, H, W, num_pos_feats]
        let pos_y = y_embed.broadcast_div(&dim_t)?;
        let pos_x = x_embed.broadcast_div(&dim_t)?;

        // Apply sin to even indices, cos to odd indices, then interleave
        let pos_y = self.apply_sin_cos_interleaved(&pos_y)?;
        let pos_x = self.apply_sin_cos_interleaved(&pos_x)?;

        // Concatenate y and x position encodings: [B, H, W, num_pos_feats] * 2 -> [B, H, W, num_pos_feats * 2]
        let pos = Tensor::cat(&[&pos_y, &pos_x], 3)?;

        // Permute to [B, C, H, W] format
        pos.permute((0, 3, 1, 2))
    }

    /// Compute cumulative sum along a dimension
    fn cumsum_2d(&self, tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = tensor.dims();
        let size = shape[dim];

        // Create indices for the cumsum
        // For dim=1 (height): we want [1, 2, 3, ..., H] repeated across width
        // For dim=2 (width): we want [1, 2, 3, ..., W] repeated across height
        let indices: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        let indices = Tensor::from_vec(indices, size, tensor.device())?;

        match dim {
            1 => {
                // Reshape to [1, H, 1] and broadcast to [B, H, W]
                let indices = indices.reshape((1, size, 1))?;
                indices.broadcast_as(shape)
            }
            2 => {
                // Reshape to [1, 1, W] and broadcast to [B, H, W]
                let indices = indices.reshape((1, 1, size))?;
                indices.broadcast_as(shape)
            }
            _ => {
                candle_core::bail!("cumsum_2d only supports dim 1 or 2")
            }
        }
    }

    /// Create the dimension tensor for frequency scaling
    ///
    /// dim_t[i] = temperature ** (2 * (i // 2) / num_pos_feats)
    fn create_dim_tensor(&self, device: &candle_core::Device) -> Result<Tensor> {
        let num_pos_feats = self.config.num_pos_feats;
        let temperature = self.config.temperature;

        let dim_values: Vec<f32> = (0..num_pos_feats)
            .map(|i| {
                let exponent = 2.0 * ((i / 2) as f64) / (num_pos_feats as f64);
                temperature.powf(exponent) as f32
            })
            .collect();

        Tensor::from_vec(dim_values, num_pos_feats, device)
    }

    /// Apply sin to even indices and cos to odd indices, then interleave
    ///
    /// Input: [B, H, W, num_pos_feats]
    /// Output: [B, H, W, num_pos_feats]
    ///
    /// For input indices 0, 1, 2, 3, 4, 5, ...
    /// Output is: sin(0), cos(1), sin(2), cos(3), sin(4), cos(5), ...
    fn apply_sin_cos_interleaved(&self, pos: &Tensor) -> Result<Tensor> {
        // Get even and odd indexed elements
        // Even indices: 0, 2, 4, ... -> apply sin
        // Odd indices: 1, 3, 5, ... -> apply cos
        let pos_even = self.gather_indices(pos, 0, 2)?; // [B, H, W, num_pos_feats/2]
        let pos_odd = self.gather_indices(pos, 1, 2)?; // [B, H, W, num_pos_feats/2]

        let pos_sin = pos_even.sin()?;
        let pos_cos = pos_odd.cos()?;

        // Interleave sin and cos results
        // Stack along new dimension, then flatten
        // [B, H, W, num_pos_feats/2] * 2 -> [B, H, W, num_pos_feats/2, 2] -> [B, H, W, num_pos_feats]
        let stacked = Tensor::stack(&[&pos_sin, &pos_cos], 4)?;
        let (b, h, w, half, _two) = stacked.dims5()?;
        stacked.reshape((b, h, w, half * 2))
    }

    /// Gather elements at indices start, start+step, start+2*step, ... along the last dimension
    fn gather_indices(&self, tensor: &Tensor, start: usize, step: usize) -> Result<Tensor> {
        let shape = tensor.dims();
        let last_dim = shape[shape.len() - 1];

        // Collect indices
        let indices: Vec<usize> = (start..last_dim).step_by(step).collect();
        let num_indices = indices.len();

        // Gather along last dimension by narrowing and concatenating
        let mut gathered = Vec::with_capacity(num_indices);
        for &idx in &indices {
            gathered.push(tensor.narrow(shape.len() - 1, idx, 1)?);
        }

        Tensor::cat(&gathered.iter().collect::<Vec<_>>(), shape.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_position_encoding_shape() {
        let device = Device::Cpu;
        let pos_enc = PositionEmbeddingSine::for_rf_detr(256);

        let output = pos_enc.forward(1, 32, 32, &device).unwrap();
        assert_eq!(output.dims(), &[1, 256, 32, 32]);
    }

    #[test]
    fn test_position_encoding_range() {
        let device = Device::Cpu;
        let pos_enc = PositionEmbeddingSine::for_rf_detr(256);

        let output = pos_enc.forward(1, 32, 32, &device).unwrap();

        // Sin and cos outputs should be in [-1, 1]
        let min = output
            .flatten_all()
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let max = output
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(min >= -1.0 - 1e-6, "min {} should be >= -1", min);
        assert!(max <= 1.0 + 1e-6, "max {} should be <= 1", max);
    }
}
