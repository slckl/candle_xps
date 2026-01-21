//! Predict subcommand for RF-DETR object detection and instance segmentation.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use clap::Args;
use image::{DynamicImage, Rgba};

use crate::coco_classes;
use crate::config::RfDetrConfig;
use crate::detection::Detection;
use crate::detection::DetectionWithMask;
use crate::detection::RawPrediction;
use crate::model::RfDetr;
use crate::preprocess;
use crate::Which;

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

/// Draw detections on an image (without masks)
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

/// Draw detections with segmentation masks on an image
fn draw_detections_with_masks(
    img: DynamicImage,
    detections: &[DetectionWithMask],
    legend_size: u32,
) -> Result<DynamicImage> {
    let (img_width, img_height) = (img.width() as usize, img.height() as usize);
    let mut img = img.to_rgba8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle_core::Error::wrap)?;

    // First pass: draw all masks with transparency
    for det_with_mask in detections {
        let det = &det_with_mask.detection;
        let box_color = get_class_color(det.class_id);

        // Draw mask if available
        if let (Some(mask), Some((_mask_h, mask_w))) =
            (&det_with_mask.mask, det_with_mask.mask_dims)
        {
            // Create semi-transparent color for mask (alpha = 100 out of 255)
            let mask_color = Rgba([box_color.0[0], box_color.0[1], box_color.0[2], 100]);

            // The mask is already resized to image dimensions via bilinear interpolation
            for y in 0..img_height {
                for x in 0..img_width {
                    let mask_idx = y * mask_w + x;

                    if mask_idx < mask.len() && mask[mask_idx] {
                        // Blend the mask color with the existing pixel
                        let pixel = img.get_pixel(x as u32, y as u32);
                        let alpha = mask_color.0[3] as f32 / 255.0;
                        let blended = Rgba([
                            ((1.0 - alpha) * pixel.0[0] as f32 + alpha * mask_color.0[0] as f32)
                                as u8,
                            ((1.0 - alpha) * pixel.0[1] as f32 + alpha * mask_color.0[1] as f32)
                                as u8,
                            ((1.0 - alpha) * pixel.0[2] as f32 + alpha * mask_color.0[2] as f32)
                                as u8,
                            255,
                        ]);
                        img.put_pixel(x as u32, y as u32, blended);
                    }
                }
            }
        }
    }

    // Convert to RGB8 for drawing boxes and text
    let mut img_rgb = DynamicImage::ImageRgba8(img).to_rgb8();

    // Second pass: draw bounding boxes and labels on top
    for det_with_mask in detections {
        let det = &det_with_mask.detection;
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
                &mut img_rgb,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, dy),
                box_color,
            );
        }

        // Draw label
        if legend_size > 0 {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img_rgb,
                imageproc::rect::Rect::at(x1, y1).of_size(dx, legend_size),
                label_bg_color,
            );
            let legend = format!("{} {:.0}%", class_name, 100.0 * det.score);
            imageproc::drawing::draw_text_mut(
                &mut img_rgb,
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

    Ok(DynamicImage::ImageRgb8(img_rgb))
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
    let (class_logits, bbox_predictions, _mask_logits) = model.forward(&batch_tensor)?;

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

/// Filter raw predictions with masks by confidence threshold and skip background class.
/// Returns indices of kept detections for mask extraction.
fn filter_detections_with_indices(
    raw: &RawPrediction,
    confidence_threshold: f32,
) -> Vec<(usize, Detection)> {
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

        detections.push((
            i,
            Detection {
                bbox: raw.boxes[i],
                score,
                class_id,
            },
        ));
    }

    // Sort by score descending
    detections.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap());

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

// TODO remove this in favor of a single predict function with/without masks flag
/// Run inference on a single image with segmentation masks.
fn predict_image_with_masks(
    model: &RfDetr,
    image_path: &str,
    config: &RfDetrConfig,
    device: &Device,
    confidence_threshold: f32,
) -> anyhow::Result<Vec<DetectionWithMask>> {
    println!("Preprocessing image...");

    // Steps 01-03: Load, normalize, and resize image
    let (preprocessed, h_orig, w_orig) =
        preprocess::preprocess_image(image_path, config.resolution, device)?;

    // Add batch dimension: [3, H, W] -> [1, 3, H, W]
    let batch_tensor = preprocess::add_batch_dim(&preprocessed)?;

    println!("  Original image size: {}x{}", w_orig, h_orig);

    // Run full model forward pass with masks
    let (class_logits, bbox_predictions, mask_logits) = model.forward(&batch_tensor)?;

    let Some(mask_logits) = mask_logits else {
        anyhow::bail!("Model did not produce segmentation masks")
    };

    // Post-process detections
    let raw_pred = postprocess_outputs(&class_logits, &bbox_predictions, h_orig, w_orig)?;

    // Filter detections and keep track of indices for mask extraction
    let filtered = filter_detections_with_indices(&raw_pred, confidence_threshold);

    println!(
        "  Found {} detections above threshold {}",
        filtered.len(),
        confidence_threshold
    );

    // Extract masks for filtered detections
    // mask_logits shape: [1, num_queries, mask_h, mask_w]
    let mask_logits = mask_logits.squeeze(0)?; // [num_queries, mask_h, mask_w]

    let mut detections_with_masks = Vec::with_capacity(filtered.len());

    for (query_idx, detection) in filtered {
        // Extract mask for this query: [mask_h, mask_w]
        let mask_tensor = mask_logits.get(query_idx)?;

        // Reshape to [1, 1, mask_h, mask_w] for bilinear interpolation
        let mask_4d = mask_tensor.unsqueeze(0)?.unsqueeze(0)?;

        // Resize mask to original image size using bilinear interpolation
        // This matches Python's F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        let mask_resized = mask_4d.upsample_bilinear2d(h_orig, w_orig, false)?;

        // Squeeze back to [h_orig, w_orig] and convert to binary mask
        let mask_flat: Vec<f32> = mask_resized.flatten_all()?.to_vec1()?;

        // Convert to binary mask by thresholding on logits > 0.0
        // This matches Python's: res_i['masks'] = masks_i > 0.0
        // (threshold on raw logits, not sigmoid)
        let binary_mask: Vec<bool> = mask_flat.iter().map(|&v| v > 0.0).collect();

        detections_with_masks.push(DetectionWithMask {
            detection,
            mask: Some(binary_mask),
            mask_dims: Some((h_orig, w_orig)),
        });
    }

    Ok(detections_with_masks)
}

/// Load model from path.
pub fn load_model(model_path: &PathBuf, config: &RfDetrConfig, device: &Device) -> Result<RfDetr> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
    RfDetr::load(vb, config)
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
    if config.segmentation_head {
        println!("  Segmentation: enabled");
    }

    // Load model weights
    println!("Loading model from: {:?}", model_path);

    let start = Instant::now();
    let model = load_model(&model_path, &config, device)?;
    println!("Model loaded successfully in {:?}", start.elapsed());

    // Load original image for annotation
    let img = image::ImageReader::open(&args.image)?.decode()?;

    let annotated = if config.segmentation_head {
        // Run inference with masks for segmentation model
        let start = Instant::now();
        let detections = predict_image_with_masks(
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

        println!("Annotating {} segmentation masks", detections.len());

        // Draw detections with masks
        draw_detections_with_masks(img, &detections, args.legend_size)?
    } else {
        // Run inference without masks for detection-only model
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

        // Draw detections
        draw_detections(img, &detections, args.legend_size)?
    };

    // Save output
    let mut output_path = PathBuf::from(&args.image);
    output_path.set_extension("pp.jpg");
    annotated.save(&output_path)?;
    println!("Annotated image saved to: {:?}", output_path);

    Ok(())
}
