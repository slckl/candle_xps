//! Multi-Scale Projector for RF-DETR Backbone
//!
//! This module implements the MultiScaleProjector which takes the multi-scale
//! feature maps from the DINOv2 backbone encoder and projects them to the
//! required output channels for the transformer decoder.
//!
//! For RF-DETR small/medium/nano models with projector_scale=["P4"] and scale_factor=1.0:
//! - Input: 4 feature maps of shape [B, 384, H, W] each
//! - Concatenated: [B, 1536, H, W]
//! - Output: 1 feature map of shape [B, 256, H, W]
//!
//! For RF-DETR large model with projector_scale=["P3", "P5"] and scale_factors=[2.0, 0.5]:
//! - Input: 4 feature maps of shape [B, 768, H, W] each
//! - P3 (scale=2.0): Each feature upsampled via ConvTranspose2d (768 -> 384), concat to 1536
//! - P5 (scale=0.5): Each feature downsampled via ConvX stride=2 (768 -> 768), concat to 3072
//!
//! The structure is:
//! - stages_sampling: List of sampling modules per scale (identity/upsample/downsample)
//! - stages: C2f -> LayerNorm

use candle_core::{Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder};

/// 2D Layer Normalization (channels-last style, applied to NCHW tensors)
///
/// This is the LayerNorm variant used in the projector, which normalizes
/// over the channel dimension for inputs of shape (batch, channels, height, width).
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    normalized_shape: usize,
}

impl LayerNorm2d {
    pub fn load(vb: VarBuilder, normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(normalized_shape, "weight")?;
        let bias = vb.get(normalized_shape, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps,
            normalized_shape,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, H, W] -> [B, H, W, C]
        let x = x.permute((0, 2, 3, 1))?;

        // Apply layer norm over last dimension (C)
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;

        // Apply weight and bias
        let x_scaled = x_normed.broadcast_mul(&self.weight)?;
        let x_biased = x_scaled.broadcast_add(&self.bias)?;

        // [B, H, W, C] -> [B, C, H, W]
        x_biased.permute((0, 3, 1, 2))
    }
}

/// Activation type for ConvX
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Silu,
    Relu,
}

/// Convolution + LayerNorm + Activation module
///
/// This is the ConvX module from the Python implementation.
pub struct ConvX {
    conv: Conv2d,
    bn: LayerNorm2d,
    activation: Activation,
}

impl ConvX {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        Self::load_with_activation(
            vb,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            Activation::Silu,
        )
    }

    pub fn load_with_activation(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        activation: Activation,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        let conv_config = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_size,
            conv_config,
            vb.pp("conv"),
        )?;
        let bn = LayerNorm2d::load(vb.pp("bn"), out_channels, 1e-6)?;

        Ok(Self {
            conv,
            bn,
            activation,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let x = self.conv.forward(&x)?;
        let x = self.bn.forward(&x)?;
        match self.activation {
            Activation::Silu => candle_nn::ops::silu(&x),
            Activation::Relu => x.relu(),
        }
    }
}

/// Simple ConvTranspose2d for upsampling (no normalization or activation)
///
/// Used in scale=2.0 sampling for the large model projector.
/// Weight path: stages_sampling.{scale_idx}.{feat_idx}.0.weight/bias
pub struct ConvTransposeSimple {
    conv: ConvTranspose2d,
}

impl ConvTransposeSimple {
    pub fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let config = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        // Weight path is just "0" since it's Sequential((0): ConvTranspose2d(...))
        let conv = candle_nn::conv_transpose2d(
            in_channels,
            out_channels,
            2, // kernel_size
            config,
            vb.pp("0"),
        )?;

        Ok(Self { conv })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        self.conv.forward(&x)
    }
}

/// Sampling module for a single feature map in the projector
///
/// For scale=1.0: Identity (no resampling)
/// For scale=2.0: ConvTranspose2d upsampling (halves channels)
/// For scale=0.5: ConvX with stride=2 downsampling (keeps channels)
pub enum SamplingModule {
    Identity,
    /// Upsample with ConvTranspose2d (scale=2.0): in_ch -> in_ch/2
    Upsample(ConvTransposeSimple),
    /// Downsample with ConvX stride=2 (scale=0.5): in_ch -> in_ch
    Downsample(ConvX),
}

impl SamplingModule {
    pub fn load_for_scale(vb: VarBuilder, in_channels: usize, scale: f64) -> Result<(Self, usize)> {
        // Returns (module, output_channels)
        if scale == 1.0 {
            Ok((SamplingModule::Identity, in_channels))
        } else if scale == 2.0 {
            // Upsample: ConvTranspose2d halves channels
            // Weight path: stages_sampling.{scale}.{feat}.0.weight/bias
            let out_channels = in_channels / 2;
            let block = ConvTransposeSimple::load(vb, in_channels, out_channels)?;
            Ok((SamplingModule::Upsample(block), out_channels))
        } else if scale == 0.5 {
            // Downsample: ConvX with stride=2, keeps channels
            // Weight path: stages_sampling.{scale}.{feat}.0.conv.weight, .0.bn.weight/bias
            let convx = ConvX::load_with_activation(
                vb.pp("0"),
                in_channels,
                in_channels,
                3,
                2,
                Activation::Relu, // Uses ReLU for downsampling
            )?;
            Ok((SamplingModule::Downsample(convx), in_channels))
        } else {
            candle_core::bail!("Unsupported scale factor: {}", scale);
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            SamplingModule::Identity => Ok(x.clone()),
            SamplingModule::Upsample(block) => block.forward(x),
            SamplingModule::Downsample(convx) => convx.forward(x),
        }
    }
}

/// Standard Bottleneck block
///
/// Consists of two ConvX layers with optional residual connection.
pub struct Bottleneck {
    cv1: ConvX,
    cv2: ConvX,
    add: bool, // Whether to use residual connection
}

impl Bottleneck {
    pub fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        // e=1.0 means hidden channels = c2
        let c_ = c2; // hidden channels (e=1.0)

        let cv1 = ConvX::load(vb.pp("cv1"), c1, c_, 3, 1)?;
        let cv2 = ConvX::load(vb.pp("cv2"), c_, c2, 3, 1)?;

        let add = shortcut && c1 == c2;

        Ok(Self { cv1, cv2, add })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.add {
            x + out
        } else {
            Ok(out)
        }
    }
}

/// C2f module - Faster Implementation of CSP Bottleneck with 2 convolutions
///
/// Structure:
/// - cv1: ConvX that outputs 2*c channels (where c = out_channels * e)
/// - Split output into two parts of c channels each
/// - Pass second part through n Bottleneck modules
/// - Concatenate all parts: [first_split, second_split, bottleneck_0_out, ..., bottleneck_n-1_out]
/// - cv2: ConvX that takes (2 + n) * c channels and outputs out_channels
pub struct C2f {
    c: usize, // hidden channels
    cv1: ConvX,
    cv2: ConvX,
    bottlenecks: Vec<Bottleneck>,
}

impl C2f {
    pub fn load(
        vb: VarBuilder,
        c1: usize,      // input channels
        c2: usize,      // output channels
        n: usize,       // number of bottlenecks
        shortcut: bool, // whether to use shortcut in bottlenecks
        e: f64,         // expansion ratio
    ) -> Result<Self> {
        let c = (c2 as f64 * e) as usize; // hidden channels

        // cv1 outputs 2*c channels
        let cv1 = ConvX::load(vb.pp("cv1"), c1, 2 * c, 1, 1)?;

        // cv2 takes (2 + n) * c channels and outputs c2
        let cv2 = ConvX::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1)?;

        // Load n bottleneck modules
        let mut bottlenecks = Vec::with_capacity(n);
        for i in 0..n {
            let bottleneck = Bottleneck::load(vb.pp(format!("m.{}", i)), c, c, shortcut)?;
            bottlenecks.push(bottleneck);
        }

        Ok(Self {
            c,
            cv1,
            cv2,
            bottlenecks,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // cv1 forward
        let cv1_out = self.cv1.forward(x)?;

        // Split into two parts of c channels each
        // Using narrow instead of split
        let y0 = cv1_out.narrow(1, 0, self.c)?;
        let y1 = cv1_out.narrow(1, self.c, self.c)?;

        let mut parts = vec![y0, y1];

        // Pass through bottlenecks, each taking the previous output
        let mut last = parts.last().unwrap().clone();
        for bottleneck in &self.bottlenecks {
            last = bottleneck.forward(&last)?;
            parts.push(last.clone());
        }

        // Concatenate all parts along channel dimension
        let y_cat = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        // cv2 forward
        self.cv2.forward(&y_cat)
    }
}

/// Configuration for MultiScaleProjector
#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    /// Input channels for each feature map
    pub in_channels: Vec<usize>,
    /// Output channels
    pub out_channels: usize,
    /// Scale factors for each output level (e.g., [1.0] for P4 only)
    pub scale_factors: Vec<f64>,
    /// Number of bottleneck blocks in C2f
    pub num_blocks: usize,
}

impl ProjectorConfig {
    /// Create projector config for RF-DETR small model (P4 only, scale=1.0)
    pub fn small(
        hidden_dim: usize,
        encoder_hidden_size: usize,
        num_encoder_outputs: usize,
    ) -> Self {
        Self {
            // [384, 384, 384, 384]
            in_channels: vec![encoder_hidden_size; num_encoder_outputs],
            // 256
            out_channels: hidden_dim,
            // P4 only
            scale_factors: vec![1.0],
            num_blocks: 3,
        }
    }

    /// Create projector config for RF-DETR large model (P3+P5, scale_factors=[2.0, 0.5])
    pub fn large(
        hidden_dim: usize,
        encoder_hidden_size: usize,
        num_encoder_outputs: usize,
    ) -> Self {
        Self {
            // [768, 768, 768, 768]
            in_channels: vec![encoder_hidden_size; num_encoder_outputs],
            // 384
            out_channels: hidden_dim,
            // P3 and P5
            scale_factors: vec![2.0, 0.5],
            num_blocks: 3,
        }
    }
}

/// Multi-Scale Projector
///
/// For RF-DETR small with scale_factor=1.0:
/// - stages_sampling is identity (no resampling)
/// - stages contains: C2f -> LayerNorm
///
/// For RF-DETR large with scale_factors=[2.0, 0.5]:
/// - stages_sampling[0] (P3): ConvTranspose2d upsampling per feature
/// - stages_sampling[1] (P5): ConvX downsample per feature
/// - stages contains: C2f -> LayerNorm for each scale
pub struct MultiScaleProjector {
    /// Scale factors for each output level
    scale_factors: Vec<f64>,
    /// Sampling modules for each scale and each input feature
    /// Outer vec: per scale, Inner vec: per input feature
    stages_sampling: Vec<Vec<SamplingModule>>,
    /// C2f module for each scale
    c2f_modules: Vec<C2f>,
    /// Final LayerNorm for each scale
    layer_norms: Vec<LayerNorm2d>,
    /// Number of input feature maps
    num_inputs: usize,
}

impl MultiScaleProjector {
    pub fn load(vb: VarBuilder, config: &ProjectorConfig) -> Result<Self> {
        let mut stages_sampling = Vec::new();
        let mut c2f_modules = Vec::new();
        let mut layer_norms = Vec::new();

        for (i, &scale) in config.scale_factors.iter().enumerate() {
            // Load sampling modules for each input feature
            let mut sampling_modules = Vec::new();
            let mut total_out_channels = 0usize;

            for (j, &in_ch) in config.in_channels.iter().enumerate() {
                let (module, out_ch) = SamplingModule::load_for_scale(
                    vb.pp(format!("stages_sampling.{}.{}", i, j)),
                    in_ch,
                    scale,
                )?;
                sampling_modules.push(module);
                total_out_channels += out_ch;
            }

            stages_sampling.push(sampling_modules);

            // Load C2f module
            // stages.{i}.0 is C2f
            let c2f = C2f::load(
                vb.pp(format!("stages.{}.0", i)),
                total_out_channels,
                config.out_channels,
                config.num_blocks,
                false, // shortcut=False in projector
                0.5,   // e=0.5
            )?;
            c2f_modules.push(c2f);

            // Load LayerNorm
            // stages.{i}.1 is LayerNorm
            let ln =
                LayerNorm2d::load(vb.pp(format!("stages.{}.1", i)), config.out_channels, 1e-6)?;
            layer_norms.push(ln);
        }

        Ok(Self {
            scale_factors: config.scale_factors.clone(),
            stages_sampling,
            c2f_modules,
            layer_norms,
            num_inputs: config.in_channels.len(),
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Vector of feature maps from encoder, each of shape [B, C, H, W]
    ///
    /// # Returns
    /// Vector of projected feature maps, each of shape [B, out_channels, H', W']
    pub fn forward(&self, x: &[Tensor]) -> Result<Vec<Tensor>> {
        assert_eq!(
            x.len(),
            self.num_inputs,
            "Expected {} input feature maps, got {}",
            self.num_inputs,
            x.len()
        );

        let mut results = Vec::new();

        for (_stage_idx, ((sampling_modules, c2f), ln)) in self
            .stages_sampling
            .iter()
            .zip(self.c2f_modules.iter())
            .zip(self.layer_norms.iter())
            .enumerate()
        {
            // Apply sampling to each input feature
            let mut sampled_features = Vec::new();
            for (feat, sampling) in x.iter().zip(sampling_modules.iter()) {
                let sampled = sampling.forward(feat)?;
                sampled_features.push(sampled);
            }

            // Concatenate all sampled features along channel dimension
            let feat_fuse = if sampled_features.len() > 1 {
                let tensors: Vec<&Tensor> = sampled_features.iter().collect();
                Tensor::cat(&tensors, 1)?
            } else {
                sampled_features.into_iter().next().unwrap()
            };

            // Apply C2f
            let out = c2f.forward(&feat_fuse)?;

            // Apply LayerNorm
            let out = ln.forward(&out)?;

            results.push(out);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;

    #[test]
    fn test_projector_config_small() {
        let config = ProjectorConfig::small(256, 384, 4);
        assert_eq!(config.in_channels, vec![384, 384, 384, 384]);
        assert_eq!(config.out_channels, 256);
        assert_eq!(config.scale_factors, vec![1.0]);
        assert_eq!(config.num_blocks, 3);
    }

    /// Integration test comparing projector output against Python reference
    ///
    /// Run with: cargo test test_projector_against_python -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_projector_against_python() {
        use candle_nn::VarBuilder;

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

        // Create projector config for small model
        let config = ProjectorConfig::small(256, 384, 4);
        let projector = MultiScaleProjector::load(vb.pp("backbone.0.projector"), &config)
            .expect("Failed to load projector");

        // Load Python's encoder outputs (step 04) as input
        let mut encoder_outputs = Vec::new();
        for i in 0..4 {
            let path = format!("{}/04_backbone_encoder_output_{}.npy", DEBUG_DIR, i);
            if std::path::Path::new(&path).exists() {
                encoder_outputs.push(load_npy(&path));
            } else {
                println!("Encoder output file not found: {}", path);
                return;
            }
        }

        println!("Loaded {} encoder outputs", encoder_outputs.len());
        for (i, t) in encoder_outputs.iter().enumerate() {
            println!("  Encoder output {}: shape={:?}", i, t.dims());
        }

        // Run projector forward
        let projector_outputs = projector
            .forward(&encoder_outputs)
            .expect("Projector forward failed");

        println!(
            "\nProjector outputs: {} feature maps",
            projector_outputs.len()
        );
        assert_eq!(projector_outputs.len(), 1, "Expected 1 output for P4 scale");

        // Compare with Python reference (step 05)
        let ref_path = format!("{}/05_backbone_projector_output_0.npy", DEBUG_DIR);
        if std::path::Path::new(&ref_path).exists() {
            let reference = load_npy(&ref_path);
            println!("Reference shape: {:?}", reference.dims());
            println!("Rust output shape: {:?}", projector_outputs[0].dims());

            compare_tensors(
                "05_backbone_projector_output_0",
                &projector_outputs[0],
                &reference,
                0.05, // Allow small floating point differences from LayerNorm/Conv implementations
            );
        } else {
            println!("Reference file not found: {}", ref_path);
            println!(
                "Run: cd py/rfdetr && uv run python3 predict_study.py -m small -d output -i sample.jpg"
            );
        }
    }
}
