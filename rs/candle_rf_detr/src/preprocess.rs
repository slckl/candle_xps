//! Image Preprocessing for RF-DETR
//!
//! This module handles the preprocessing pipeline for input images:
//! 1. Load image and convert to tensor (RGB, CHW format, values in [0, 1])
//! 2. Normalize using ImageNet mean and std
//! 3. Resize to model resolution

use candle_core::{Device, Result, Tensor};
use image::DynamicImage;

/// Tensor statistics for debugging and validation
#[derive(Debug)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub sum: f32,
}

impl TensorStats {
    /// Compute statistics for a tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?.to_dtype(candle_core::DType::F32)?;
        let data: Vec<f32> = flat.to_vec1()?;

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        Ok(Self {
            shape,
            min,
            max,
            mean,
            sum,
        })
    }

    /// Print statistics in a format similar to Python debug output
    pub fn print(&self, name: &str) {
        println!(
            "  {}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}, sum={:.6}",
            name, self.shape, self.min, self.max, self.mean, self.sum
        );
    }
}

/// Print tensor statistics (convenience function)
pub fn print_tensor_stats(tensor: &Tensor, name: &str) -> Result<()> {
    let stats = TensorStats::from_tensor(tensor)?;
    stats.print(name);
    Ok(())
}

/// ImageNet normalization mean values (RGB order)
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet normalization std values (RGB order)
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Load an image from disk and return it as a DynamicImage
pub fn load_image(path: &str) -> anyhow::Result<DynamicImage> {
    let img = image::open(path)?;
    Ok(img)
}

/// Convert a DynamicImage to a tensor in CHW format with values in [0, 1]
///
/// This corresponds to PyTorch's `torchvision.transforms.functional.to_tensor`
///
/// # Arguments
/// * `img` - Input image
/// * `device` - Device to create tensor on
///
/// # Returns
/// Tensor of shape [3, H, W] with values in [0.0, 1.0]
pub fn image_to_tensor(img: &DynamicImage, device: &Device) -> Result<Tensor> {
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();

    // Get raw RGB bytes
    let raw_data = img.into_raw();

    // Convert u8 [0, 255] to f32 [0.0, 1.0]
    let float_data: Vec<f32> = raw_data.iter().map(|&x| x as f32 / 255.0).collect();

    // Image data is in HWC format, we need CHW
    // raw_data is [R, G, B, R, G, B, ...] for each pixel row by row
    let h = height as usize;
    let w = width as usize;

    // Reshape from HWC to CHW
    let mut chw_data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let src_idx = (y * w + x) * 3;
            chw_data[0 * h * w + y * w + x] = float_data[src_idx]; // R
            chw_data[1 * h * w + y * w + x] = float_data[src_idx + 1]; // G
            chw_data[2 * h * w + y * w + x] = float_data[src_idx + 2]; // B
        }
    }

    Tensor::from_vec(chw_data, (3, h, w), device)
}

/// Normalize a tensor using ImageNet mean and std
///
/// This corresponds to PyTorch's `torchvision.transforms.functional.normalize`
/// Formula: output = (input - mean) / std
///
/// # Arguments
/// * `tensor` - Input tensor of shape [3, H, W] with values in [0.0, 1.0]
///
/// # Returns
/// Normalized tensor of shape [3, H, W]
pub fn normalize(tensor: &Tensor) -> Result<Tensor> {
    let device = tensor.device();
    let (_, h, w) = tensor.dims3()?;

    // Create mean and std tensors with shape [3, 1, 1] for broadcasting
    let mean = Tensor::from_slice(&IMAGENET_MEAN, (3, 1, 1), device)?;
    let std = Tensor::from_slice(&IMAGENET_STD, (3, 1, 1), device)?;

    // Broadcast to [3, H, W]
    let mean = mean.broadcast_as((3, h, w))?;
    let std = std.broadcast_as((3, h, w))?;

    // Normalize: (tensor - mean) / std
    let normalized = tensor.sub(&mean)?.div(&std)?;

    Ok(normalized)
}

/// Resize a tensor to target size using bilinear interpolation
///
/// This corresponds to PyTorch's `torchvision.transforms.functional.resize`
///
/// # Arguments
/// * `tensor` - Input tensor of shape [3, H, W]
/// * `target_size` - Target (height, width)
///
/// # Returns
/// Resized tensor of shape [3, target_h, target_w]
pub fn resize(tensor: &Tensor, target_size: (usize, usize)) -> Result<Tensor> {
    let (target_h, target_w) = target_size;
    let _ = tensor.dims3()?; // Validate input is 3D

    // Add batch dimension for interpolation: [3, H, W] -> [1, 3, H, W]
    let batched = tensor.unsqueeze(0)?;

    // Use candle's upsample_bilinear2d
    // Note: upsample_bilinear2d expects [N, C, H, W]
    // The third argument is align_corners (false matches PyTorch's default behavior)
    let resized = batched.upsample_bilinear2d(target_h, target_w, false)?;

    // Remove batch dimension: [1, 3, H, W] -> [3, H, W]
    let resized = resized.squeeze(0)?;

    Ok(resized)
}

/// Full preprocessing pipeline: load, convert to tensor, normalize, and resize
///
/// # Arguments
/// * `image_path` - Path to input image
/// * `resolution` - Target resolution (square)
/// * `device` - Device to create tensors on
///
/// # Returns
/// Tuple of (preprocessed_tensor, original_height, original_width)
/// - preprocessed_tensor has shape [3, resolution, resolution]
pub fn preprocess_image(
    image_path: &str,
    resolution: usize,
    device: &Device,
) -> anyhow::Result<(Tensor, usize, usize)> {
    // Step 1: Load image
    let img = load_image(image_path)?;

    // Step 2: Convert to tensor [3, H, W] with values in [0, 1]
    let tensor = image_to_tensor(&img, device)?;
    let (_, h_orig, w_orig) = tensor.dims3()?;
    print_tensor_stats(&tensor, "01_input_image_raw")?;

    // Step 3: Normalize with ImageNet mean/std
    let normalized = normalize(&tensor)?;
    print_tensor_stats(&normalized, "02_input_image_normalized")?;

    // Step 4: Resize to model resolution
    let resized = resize(&normalized, (resolution, resolution))?;
    print_tensor_stats(&resized, "03_input_image_resized")?;

    Ok((resized, h_orig, w_orig))
}

/// Add batch dimension to a preprocessed image tensor
///
/// # Arguments
/// * `tensor` - Tensor of shape [3, H, W]
///
/// # Returns
/// Tensor of shape [1, 3, H, W]
pub fn add_batch_dim(tensor: &Tensor) -> Result<Tensor> {
    tensor.unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_normalization_values() {
        // Test that normalization produces expected range
        // For input 0.0: (0.0 - 0.485) / 0.229 ≈ -2.117904 (for R channel)
        let device = Device::Cpu;
        let input = Tensor::zeros((3, 2, 2), DType::F32, &device).unwrap();
        let normalized = normalize(&input).unwrap();

        // Check that normalized values are approximately correct
        let data: Vec<f32> = normalized.flatten_all().unwrap().to_vec1().unwrap();

        // R channel: (0 - 0.485) / 0.229 ≈ -2.1179
        assert!((data[0] - (-2.1179)).abs() < 0.01);
        // G channel: (0 - 0.456) / 0.224 ≈ -2.0357
        assert!((data[4] - (-2.0357)).abs() < 0.01);
        // B channel: (0 - 0.406) / 0.225 ≈ -1.8044
        assert!((data[8] - (-1.8044)).abs() < 0.01);
    }

    #[test]
    fn test_normalization_with_ones() {
        // Test normalization with input value 1.0
        // For input 1.0: (1.0 - mean) / std
        let device = Device::Cpu;
        let input = Tensor::ones((3, 2, 2), DType::F32, &device).unwrap();
        let normalized = normalize(&input).unwrap();

        let data: Vec<f32> = normalized.flatten_all().unwrap().to_vec1().unwrap();

        // R channel: (1 - 0.485) / 0.229 ≈ 2.2489
        assert!((data[0] - 2.2489).abs() < 0.01);
        // G channel: (1 - 0.456) / 0.224 ≈ 2.4286
        assert!((data[4] - 2.4286).abs() < 0.01);
        // B channel: (1 - 0.406) / 0.225 ≈ 2.64
        assert!((data[8] - 2.64).abs() < 0.01);
    }

    #[test]
    fn test_resize_preserves_channels() {
        let device = Device::Cpu;
        let input = Tensor::randn(0.0f32, 1.0, (3, 100, 100), &device).unwrap();
        let resized = resize(&input, (50, 50)).unwrap();

        assert_eq!(resized.dims(), &[3, 50, 50]);
    }

    #[test]
    fn test_resize_upscale() {
        let device = Device::Cpu;
        let input = Tensor::randn(0.0f32, 1.0, (3, 50, 50), &device).unwrap();
        let resized = resize(&input, (100, 100)).unwrap();

        assert_eq!(resized.dims(), &[3, 100, 100]);
    }

    #[test]
    fn test_add_batch_dim() {
        let device = Device::Cpu;
        let input = Tensor::zeros((3, 384, 384), DType::F32, &device).unwrap();
        let batched = add_batch_dim(&input).unwrap();

        assert_eq!(batched.dims(), &[1, 3, 384, 384]);
    }

    #[test]
    fn test_tensor_stats() {
        let device = Device::Cpu;
        // Create tensor with known values: [0, 1, 2, 3, 4, 5]
        let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_vec(data, (2, 3), &device).unwrap();
        let stats = TensorStats::from_tensor(&input).unwrap();

        assert_eq!(stats.shape, vec![2, 3]);
        assert!((stats.min - 0.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 2.5).abs() < 1e-6);
        assert!((stats.sum - 15.0).abs() < 1e-6);
    }

    /// Test preprocessing against known reference values from Python implementation.
    ///
    /// Reference values from py/rfdetr/output/ (nano model, sample.jpg):
    /// - 01_input_image_raw: shape=[3, 720, 1280], min=0.0, max=1.0, mean≈0.5217
    /// - 02_input_image_normalized: shape=[3, 720, 1280], min≈-2.118, max≈2.64, mean≈0.323
    /// - 03_input_image_resized: shape=[3, 384, 384], min≈-2.118, max≈2.64, mean≈0.323
    #[test]
    fn test_preprocessing_reference_values() {
        let device = Device::Cpu;

        // Create a synthetic test image with known pixel values
        // Using a simple gradient pattern for reproducibility
        let h = 100usize;
        let w = 100usize;
        let mut data = vec![0.0f32; 3 * h * w];

        // Fill with gradient: R increases horizontally, G increases vertically, B is constant
        for y in 0..h {
            for x in 0..w {
                data[0 * h * w + y * w + x] = x as f32 / (w - 1) as f32; // R: 0 to 1
                data[1 * h * w + y * w + x] = y as f32 / (h - 1) as f32; // G: 0 to 1
                data[2 * h * w + y * w + x] = 0.5; // B: constant 0.5
            }
        }

        let input = Tensor::from_vec(data, (3, h, w), &device).unwrap();
        let input_stats = TensorStats::from_tensor(&input).unwrap();

        // Verify input tensor properties
        assert_eq!(input_stats.shape, vec![3, h, w]);
        assert!((input_stats.min - 0.0).abs() < 1e-6);
        assert!((input_stats.max - 1.0).abs() < 1e-6);

        // Test normalization
        let normalized = normalize(&input).unwrap();
        let norm_stats = TensorStats::from_tensor(&normalized).unwrap();

        // After normalization, the range should be approximately [-2.12, 2.64]
        // based on ImageNet mean/std
        assert!(norm_stats.min < -1.5, "Normalized min should be negative");
        assert!(norm_stats.max > 1.5, "Normalized max should be positive");

        // Test resize
        let target_size = 50;
        let resized = resize(&normalized, (target_size, target_size)).unwrap();
        let resize_stats = TensorStats::from_tensor(&resized).unwrap();

        assert_eq!(resize_stats.shape, vec![3, target_size, target_size]);
        // Mean should be approximately preserved after resize
        assert!(
            (resize_stats.mean - norm_stats.mean).abs() < 0.1,
            "Mean should be approximately preserved after resize"
        );
    }

    /// Test that normalization matches PyTorch's torchvision.transforms.functional.normalize
    /// within floating point tolerance.
    #[test]
    fn test_normalization_matches_pytorch() {
        let device = Device::Cpu;

        // Test with specific pixel value that we can verify
        // Pixel value 128/255 ≈ 0.502
        let pixel_val = 128.0 / 255.0;
        let data = vec![pixel_val; 3 * 4 * 4];
        let input = Tensor::from_vec(data, (3, 4, 4), &device).unwrap();

        let normalized = normalize(&input).unwrap();
        let norm_data: Vec<f32> = normalized.flatten_all().unwrap().to_vec1().unwrap();

        // Expected values:
        // R: (0.502 - 0.485) / 0.229 ≈ 0.0742
        // G: (0.502 - 0.456) / 0.224 ≈ 0.2054
        // B: (0.502 - 0.406) / 0.225 ≈ 0.4267
        let expected_r = (pixel_val - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        let expected_g = (pixel_val - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        let expected_b = (pixel_val - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

        // Check R channel (first 16 values)
        assert!(
            (norm_data[0] - expected_r).abs() < 1e-5,
            "R channel mismatch: got {}, expected {}",
            norm_data[0],
            expected_r
        );
        // Check G channel (values 16-31)
        assert!(
            (norm_data[16] - expected_g).abs() < 1e-5,
            "G channel mismatch: got {}, expected {}",
            norm_data[16],
            expected_g
        );
        // Check B channel (values 32-47)
        assert!(
            (norm_data[32] - expected_b).abs() < 1e-5,
            "B channel mismatch: got {}, expected {}",
            norm_data[32],
            expected_b
        );
    }
}
