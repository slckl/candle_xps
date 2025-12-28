//! RF-DETR Object Detection with Candle
//!
//! This binary provides inference capabilities for RF-DETR models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod coco_classes;
mod config;
mod detection;
mod dino2;
mod model;
mod pos_enc;
mod preprocess;
mod projector;

use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use image::DynamicImage;

use crate::{config::RfDetrConfig, detection::Detection, model::RfDetr, preprocess::TensorStats};

/// Select the compute device
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// RF-DETR model variants
#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    Nano,
    Small,
    Medium,
    Base,
    Large,
}

impl Which {
    fn config(&self) -> RfDetrConfig {
        match self {
            Which::Nano => RfDetrConfig::nano(),
            Which::Small => RfDetrConfig::small(),
            Which::Medium => RfDetrConfig::medium(),
            Which::Base => RfDetrConfig::base(),
            Which::Large => RfDetrConfig::large(),
        }
    }

    fn default_weights(&self) -> &'static str {
        match self {
            Which::Nano => "rf-detr-nano.safetensors",
            Which::Small => "rf-detr-small.safetensors",
            Which::Medium => "rf-detr-medium.safetensors",
            Which::Base => "rf-detr-base.safetensors",
            Which::Large => "rf-detr-large.safetensors",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Path to model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::Small)]
    which: Which,

    /// Input image to process.
    image: String,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.5)]
    confidence_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

impl Args {
    fn model_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                // Try to use HuggingFace Hub to download the model
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("roboflow/rf-detr".to_string());
                let filename = self.which.default_weights();
                match api.get(filename) {
                    Ok(path) => path,
                    Err(_) => {
                        // Fall back to local file
                        std::path::PathBuf::from(filename)
                    }
                }
            }
        };
        Ok(path)
    }
}

/// Draw detections on an image
fn draw_detections(
    img: DynamicImage,
    detections: &[Detection],
    legend_size: u32,
) -> Result<DynamicImage> {
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle_core::Error::wrap)?;

    for det in detections {
        let class_name = coco_classes::get_class_name(det.class_id);
        let [x1, y1, x2, y2] = det.bbox;

        println!(
            "{}: ({:.1}, {:.1}, {:.1}, {:.1}) conf: {:.2}",
            class_name, x1, y1, x2, y2, det.score
        );

        let x1 = x1 as i32;
        let y1 = y1 as i32;
        let dx = (x2 - det.bbox[0]).max(0.0) as u32;
        let dy = (y2 - det.bbox[1]).max(0.0) as u32;

        // Draw bounding box
        if dx > 0 && dy > 0 {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, dy),
                image::Rgb([255, 0, 0]),
            );
        }

        // Draw label
        if legend_size > 0 {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, legend_size),
                image::Rgb([170, 0, 0]),
            );
            let legend = format!("{} {:.0}%", class_name, 100.0 * det.score);
            imageproc::drawing::draw_text_mut(
                &mut img,
                image::Rgb([255, 255, 255]),
                x1,
                y1,
                ab_glyph::PxScale {
                    x: legend_size as f32 - 1.0,
                    y: legend_size as f32 - 1.0,
                },
                &font,
                &legend,
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(img))
}

/// Run inference on a single image
fn predict(
    model: &RfDetr,
    image_path: &str,
    config: &RfDetrConfig,
    device: &Device,
) -> anyhow::Result<Vec<Detection>> {
    println!("Preprocessing image...");

    // Steps 01-03: Load, normalize, and resize image
    let (preprocessed, h_orig, w_orig) =
        preprocess::preprocess_image(image_path, config.resolution, device)?;

    // Add batch dimension: [3, H, W] -> [1, 3, H, W]
    let batch_tensor = preprocess::add_batch_dim(&preprocessed)?;
    println!("  Batch tensor shape: {:?}", batch_tensor.dims());
    println!("  Original image size: {}x{}", w_orig, h_orig);

    // Step 04: Run backbone encoder
    println!("Running backbone encoder...");
    let encoder_outputs = model.backbone_encoder_forward(&batch_tensor)?;

    println!(
        "  Backbone encoder outputs: {} feature maps",
        encoder_outputs.len()
    );
    for (i, feat) in encoder_outputs.iter().enumerate() {
        let stats = TensorStats::from_tensor(feat)?;
        println!("  Output {}: shape={:?}", i, stats.shape);
        stats.print(&format!("    04_backbone_encoder_output_{}", i));
    }

    // Step 05: Run projector
    println!("Running projector...");
    let projector_outputs = model.projector_forward(&encoder_outputs)?;

    println!(
        "  Projector outputs: {} feature maps",
        projector_outputs.len()
    );
    for (i, feat) in projector_outputs.iter().enumerate() {
        let stats = TensorStats::from_tensor(feat)?;
        println!("  Output {}: shape={:?}", i, stats.shape);
        stats.print(&format!("    05_backbone_projector_output_{}", i));
    }

    // Step 06: Compute position encodings
    println!("Computing position encodings...");
    let position_encodings = model.compute_position_encodings(&projector_outputs, device)?;

    println!("  Position encodings: {} tensors", position_encodings.len());
    for (i, pos) in position_encodings.iter().enumerate() {
        let stats = TensorStats::from_tensor(pos)?;
        println!("  Position encoding {}: shape={:?}", i, stats.shape);
        stats.print(&format!("    06_position_encoding_{}", i));
    }

    // TODO: Steps 07+: Run through rest of model
    todo!("Full model inference not yet implemented")
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Get device
    let device = device(args.cpu)?;
    println!("Using device: {:?}", device);

    // Get model config
    let config = args.which.config();
    println!("Model config: {:?}", args.which);
    println!("  Resolution: {}", config.resolution);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Decoder layers: {}", config.dec_layers);

    // Load model weights
    let model_path = args.model_path()?;
    println!("Loading model from: {:?}", model_path);

    if !model_path.exists() {
        anyhow::bail!(
            "Model weights not found at {:?}. Please provide a valid model path with --model, \
            or ensure the model file exists.\n\
            You may need to export the PyTorch model to safetensors format first.",
            model_path
        );
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    println!("Model weights loaded into VarBuilder");
    let model = RfDetr::load(vb, &config)?;
    println!("Model loaded successfully");

    let _detections = predict(&model, &args.image, &config, &device)?;
    todo!("annotation");
}
