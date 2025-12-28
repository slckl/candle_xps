//! RF-DETR Object Detection with Candle
//!
//! This binary provides inference capabilities for RF-DETR models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod coco_classes;
mod model;

use candle_core as candle;
use model::{nms, postprocess, Detection, RFDETRConfig, RFDETR};

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use image::DynamicImage;

use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

/// Select the compute device
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
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
    fn config(&self) -> RFDETRConfig {
        match self {
            Which::Nano => RFDETRConfig::nano(),
            Which::Small => RFDETRConfig::small(),
            Which::Medium => RFDETRConfig::medium(),
            Which::Base => RFDETRConfig::base(),
            Which::Large => RFDETRConfig::large(),
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

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::Medium)]
    which: Which,

    /// Input images to process.
    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.5)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.5)]
    nms_threshold: f32,

    /// Number of top detections to return.
    #[arg(long, default_value_t = 300)]
    num_select: usize,

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

/// Image normalization constants (ImageNet statistics)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess an image for RF-DETR inference
fn preprocess_image(img: &DynamicImage, resolution: usize, device: &Device) -> Result<Tensor> {
    // Resize to target resolution
    let img = img.resize_exact(
        resolution as u32,
        resolution as u32,
        image::imageops::FilterType::Triangle,
    );

    // Convert to RGB and normalize
    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let data = img.into_raw();

    // Create tensor [H, W, C]
    let tensor = Tensor::from_vec(data, (height, width, 3), device)?;

    // Convert to [C, H, W] and normalize to [0, 1]
    let tensor = tensor.permute((2, 0, 1))?.to_dtype(DType::F32)?;
    let tensor = (tensor / 255.0)?;

    // Apply ImageNet normalization
    let mean = Tensor::from_vec(IMAGENET_MEAN.to_vec(), (3, 1, 1), device)?.to_dtype(DType::F32)?;
    let std = Tensor::from_vec(IMAGENET_STD.to_vec(), (3, 1, 1), device)?.to_dtype(DType::F32)?;
    let tensor = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

    // Add batch dimension [1, C, H, W]
    tensor.unsqueeze(0)
}

/// Draw detections on an image
fn draw_detections(
    img: DynamicImage,
    detections: &[Detection],
    legend_size: u32,
) -> Result<DynamicImage> {
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle::Error::wrap)?;

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
fn run_inference(
    model: &RFDETR,
    image_path: &str,
    args: &Args,
    device: &Device,
) -> anyhow::Result<()> {
    let mut image_path = std::path::PathBuf::from(image_path);
    println!("Processing: {:?}", image_path);

    // Load and preprocess image
    let original_image = image::ImageReader::open(&image_path)?
        .decode()
        .map_err(candle::Error::wrap)?;

    let (orig_h, orig_w) = (
        original_image.height() as usize,
        original_image.width() as usize,
    );
    let resolution = model.config.resolution;

    let input = preprocess_image(&original_image, resolution, device)?;
    println!("Input shape: {:?}", input.shape());

    // Run inference
    let (pred_logits, pred_boxes) = model.forward(&input)?;
    println!(
        "Output shapes - logits: {:?}, boxes: {:?}",
        pred_logits.shape(),
        pred_boxes.shape()
    );

    // Post-process
    let mut detections = postprocess(
        &pred_logits,
        &pred_boxes,
        (orig_h, orig_w),
        args.num_select,
        args.confidence_threshold,
    )?;

    // Apply NMS
    nms(&mut detections, args.nms_threshold);

    println!("Found {} detections after NMS", detections.len());

    // Draw and save
    let output_image = draw_detections(original_image, &detections, args.legend_size)?;
    image_path.set_extension("pp.jpg");
    println!("Saving to: {:?}", image_path);
    output_image.save(&image_path)?;

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Setup tracing if enabled
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

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
    let model = RFDETR::load(vb, config)?;
    println!("Model loaded successfully");

    // Process images
    if args.images.is_empty() {
        println!("No images provided. Use: candle_rf_detr [OPTIONS] <IMAGE>...");
        return Ok(());
    }

    for image_path in &args.images {
        if let Err(e) = run_inference(&model, image_path, &args, &device) {
            eprintln!("Error processing {}: {}", image_path, e);
        }
    }

    Ok(())
}
