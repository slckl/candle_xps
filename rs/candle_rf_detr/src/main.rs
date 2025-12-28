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
mod query_embed;
mod transformer;

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
    confidence_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    println!("Preprocessing image...");

    // Steps 01-03: Load, normalize, and resize image
    let (preprocessed, h_orig, w_orig) =
        preprocess::preprocess_image(image_path, config.resolution, device)?;

    // Add batch dimension: [3, H, W] -> [1, 3, H, W]
    let batch_tensor = preprocess::add_batch_dim(&preprocessed)?;
    println!("  Batch tensor shape: {:?}", batch_tensor.dims());
    println!("  Original image size: {}x{}", w_orig, h_orig);

    // Run full model forward pass
    println!("Running model inference...");
    let (class_logits, bbox_predictions) = model.forward(&batch_tensor)?;

    println!("  class_logits shape: {:?}", class_logits.dims());
    println!("  bbox_predictions shape: {:?}", bbox_predictions.dims());

    // Post-process: convert to detections
    println!("Post-processing...");

    // Apply sigmoid to class logits to get probabilities
    let class_probs = candle_nn::ops::sigmoid(&class_logits)?;

    // Get max class and score for each query
    let (num_queries, num_classes) = (class_probs.dim(1)?, class_probs.dim(2)?);

    // Squeeze batch dimension
    let class_probs = class_probs.squeeze(0)?; // [num_queries, num_classes]
    let bbox_predictions = bbox_predictions.squeeze(0)?; // [num_queries, 4]

    // Get max scores and class ids
    let max_scores = class_probs.max(1)?; // [num_queries]
    let max_class_ids = class_probs.argmax(1)?; // [num_queries]

    // Convert to vectors
    let scores: Vec<f32> = max_scores.to_vec1()?;
    let class_ids: Vec<u32> = max_class_ids.to_vec1()?;
    let bboxes: Vec<f32> = bbox_predictions.flatten_all()?.to_vec1()?;

    // Convert boxes from (cx, cy, w, h) normalized to (x1, y1, x2, y2) pixel coordinates
    let mut detections = Vec::new();
    for i in 0..num_queries {
        let score = scores[i];
        if score < confidence_threshold {
            continue;
        }

        let class_id = class_ids[i] as usize;
        // Skip background class (class 0 in COCO is typically background)
        if class_id == 0 {
            continue;
        }

        let cx = bboxes[i * 4] * w_orig as f32;
        let cy = bboxes[i * 4 + 1] * h_orig as f32;
        let w = bboxes[i * 4 + 2] * w_orig as f32;
        let h = bboxes[i * 4 + 3] * h_orig as f32;

        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        detections.push(Detection {
            bbox: [x1, y1, x2, y2],
            score,
            class_id,
        });
    }

    // Sort by score descending
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    println!(
        "  Found {} detections above threshold {}",
        detections.len(),
        confidence_threshold
    );

    Ok(detections)
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
    println!("  Num classes: {}", config.num_classes);

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

    let detections = predict(
        &model,
        &args.image,
        &config,
        &device,
        args.confidence_threshold,
    )?;

    if detections.is_empty() {
        println!("No detections found.");
        return Ok(());
    }

    // Load original image for annotation
    let img = image::ImageReader::open(&args.image)?.decode()?;

    // Draw detections
    let annotated = draw_detections(img, &detections, args.legend_size)?;

    // Save output
    let output_path = format!("{}.out.jpg", args.image);
    annotated.save(&output_path)?;
    println!("Annotated image saved to: {}", output_path);

    Ok(())
}
