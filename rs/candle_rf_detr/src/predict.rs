//! Predict subcommand for RF-DETR object detection.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use clap::Args;
use image::DynamicImage;

use crate::coco_classes;
use crate::config::RfDetrConfig;
use crate::detection::Detection;
use crate::model::RfDetr;
use crate::preprocess;
use crate::Which;

/// Raw prediction output before thresholding.
/// Contains all 300 queries with their scores, labels, and boxes.
#[derive(Debug, Clone)]
pub struct RawPrediction {
    /// Confidence scores for each query (max across classes)
    pub scores: Vec<f32>,
    /// Class labels for each query (argmax across classes)
    pub labels: Vec<i64>,
    /// Bounding boxes in [x1, y1, x2, y2] pixel coordinates
    pub boxes: Vec<[f32; 4]>,
}

/// Arguments for the predict subcommand
#[derive(Args, Debug)]
pub struct PredictArgs {
    /// Input image to process.
    pub image: String,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.5)]
    pub confidence_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 20)]
    pub legend_size: u32,
}

/// Color palette for different classes (similar to supervision library's default palette)
const CLASS_COLORS: [[u8; 3]; 20] = [
    [255, 64, 64],   // red
    [255, 161, 54],  // orange
    [255, 221, 51],  // yellow
    [170, 255, 50],  // lime
    [50, 255, 50],   // green
    [50, 255, 170],  // mint
    [50, 255, 255],  // cyan
    [50, 170, 255],  // sky blue
    [50, 50, 255],   // blue
    [161, 50, 255],  // purple
    [255, 50, 255],  // magenta
    [255, 50, 161],  // pink
    [128, 128, 255], // light blue
    [255, 128, 128], // light red
    [128, 255, 128], // light green
    [255, 255, 128], // light yellow
    [128, 255, 255], // light cyan
    [255, 128, 255], // light magenta
    [192, 192, 192], // silver
    [255, 200, 100], // peach
];

/// Get a color for a given class ID
fn get_class_color(class_id: usize) -> image::Rgb<u8> {
    let color = CLASS_COLORS[class_id % CLASS_COLORS.len()];
    image::Rgb(color)
}

/// Get a darker version of a color for the label background
fn get_darker_color(color: image::Rgb<u8>) -> image::Rgb<u8> {
    image::Rgb([
        (color.0[0] as u16 * 2 / 3) as u8,
        (color.0[1] as u16 * 2 / 3) as u8,
        (color.0[2] as u16 * 2 / 3) as u8,
    ])
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

        // Get color based on class
        let box_color = get_class_color(det.class_id);
        let label_bg_color = get_darker_color(box_color);

        // Draw bounding box
        if dx > 0 && dy > 0 {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, dy),
                box_color,
            );
        }

        // Draw label
        if legend_size > 0 {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, legend_size),
                label_bg_color,
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

/// Post-process model outputs into raw predictions.
/// Returns all queries (typically 300) without any filtering.
pub fn postprocess_outputs(
    class_logits: &Tensor,
    bbox_predictions: &Tensor,
    h_orig: usize,
    w_orig: usize,
) -> anyhow::Result<RawPrediction> {
    // Apply sigmoid to class logits to get probabilities
    let class_probs = candle_nn::ops::sigmoid(class_logits)?;

    // Get max class and score for each query
    let num_queries = class_probs.dim(1)?;

    // Squeeze batch dimension
    let class_probs = class_probs.squeeze(0)?; // [num_queries, num_classes]
    let bbox_preds = bbox_predictions.squeeze(0)?; // [num_queries, 4]

    // Get max scores and class ids
    let max_scores = class_probs.max(1)?; // [num_queries]
    let max_class_ids = class_probs.argmax(1)?; // [num_queries]

    // Convert to vectors
    let scores: Vec<f32> = max_scores.to_vec1()?;
    let class_ids: Vec<u32> = max_class_ids.to_vec1()?;
    let bboxes: Vec<f32> = bbox_preds.flatten_all()?.to_vec1()?;

    // Convert boxes from (cx, cy, w, h) normalized to (x1, y1, x2, y2) pixel coordinates
    let mut result_scores = Vec::with_capacity(num_queries);
    let mut result_labels = Vec::with_capacity(num_queries);
    let mut result_boxes = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let score = scores[i];
        let class_id = class_ids[i] as i64;

        let cx = bboxes[i * 4] * w_orig as f32;
        let cy = bboxes[i * 4 + 1] * h_orig as f32;
        let w = bboxes[i * 4 + 2] * w_orig as f32;
        let h = bboxes[i * 4 + 3] * h_orig as f32;

        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;

        result_scores.push(score);
        result_labels.push(class_id);
        result_boxes.push([x1, y1, x2, y2]);
    }

    Ok(RawPrediction {
        scores: result_scores,
        labels: result_labels,
        boxes: result_boxes,
    })
}

/// Run inference on a single image and return raw predictions (all queries).
/// This is the core inference function used by both predict and eval.
pub fn predict_image_raw(
    model: &RfDetr,
    image_path: &str,
    config: &RfDetrConfig,
    device: &Device,
) -> anyhow::Result<(RawPrediction, usize, usize)> {
    // Steps 01-03: Load, normalize, and resize image
    let (preprocessed, h_orig, w_orig) =
        preprocess::preprocess_image(image_path, config.resolution, device)?;

    // Add batch dimension: [3, H, W] -> [1, 3, H, W]
    let batch_tensor = preprocess::add_batch_dim(&preprocessed)?;

    // Run full model forward pass
    let (class_logits, bbox_predictions) = model.forward(&batch_tensor)?;

    // Post-process
    let raw_pred = postprocess_outputs(&class_logits, &bbox_predictions, h_orig, w_orig)?;

    Ok((raw_pred, h_orig, w_orig))
}

/// Filter raw predictions by confidence threshold and skip background class.
pub fn filter_detections(raw: &RawPrediction, confidence_threshold: f32) -> Vec<Detection> {
    let mut detections = Vec::new();

    for i in 0..raw.scores.len() {
        let score = raw.scores[i];
        if score < confidence_threshold {
            continue;
        }

        let class_id = raw.labels[i] as usize;
        // Skip background class (class 0 in COCO is typically background)
        if class_id == 0 {
            continue;
        }

        detections.push(Detection {
            bbox: raw.boxes[i],
            score,
            class_id,
        });
    }

    // Sort by score descending
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    detections
}

/// Run inference on a single image with filtering.
fn predict_image(
    model: &RfDetr,
    image_path: &str,
    config: &RfDetrConfig,
    device: &Device,
    confidence_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    println!("Preprocessing image...");

    let (raw_pred, h_orig, w_orig) = predict_image_raw(model, image_path, config, device)?;

    println!("  Original image size: {}x{}", w_orig, h_orig);

    let detections = filter_detections(&raw_pred, confidence_threshold);

    println!(
        "  Found {} detections above threshold {}",
        detections.len(),
        confidence_threshold
    );

    Ok(detections)
}

/// Load model from path.
pub fn load_model(
    model_path: &PathBuf,
    config: &RfDetrConfig,
    device: &Device,
) -> anyhow::Result<RfDetr> {
    if !model_path.exists() {
        anyhow::bail!(
            "Model weights not found at {:?}. Please provide a valid model path with --model, \
            or ensure the model file exists.\n\
            You may need to export the PyTorch model to safetensors format first.",
            model_path
        );
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
    let model = RfDetr::load(vb, config)?;
    Ok(model)
}

/// Run the predict subcommand
pub fn run(
    args: &PredictArgs,
    which: Which,
    model_path: PathBuf,
    device: &Device,
) -> anyhow::Result<()> {
    // Get model config
    let config = which.config();
    println!("Model config: {:?}", which);
    println!("  Resolution: {}", config.resolution);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Decoder layers: {}", config.dec_layers);
    println!("  Num classes: {}", config.num_classes);

    // Load model weights
    println!("Loading model from: {:?}", model_path);

    let start = Instant::now();
    let model = load_model(&model_path, &config, device)?;
    println!("Model loaded successfully in {:?}", start.elapsed());

    let start = Instant::now();
    let detections = predict_image(
        &model,
        &args.image,
        &config,
        device,
        args.confidence_threshold,
    )?;
    println!("Inference completed in {:?}", start.elapsed());

    if detections.is_empty() {
        println!("No detections found.");
        return Ok(());
    }

    // Load original image for annotation
    let img = image::ImageReader::open(&args.image)?.decode()?;

    // Draw detections
    let annotated = draw_detections(img, &detections, args.legend_size)?;

    // Save output
    let mut output_path = PathBuf::from(&args.image);
    output_path.set_extension("pp.jpg");
    annotated.save(&output_path)?;
    println!("Annotated image saved to: {:?}", output_path);

    Ok(())
}
