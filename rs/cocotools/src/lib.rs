//! COCO evaluation module.
//!
//! This module provides functionality to:
//! - Load/save predictions in the same JSON format as the Python eval_study.py
//! - Load COCO ground truth annotations
//! - Compute COCO evaluation metrics (AP, AR at various IoU thresholds)

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Prediction for a single image, matching the Python JSON format.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ImagePrediction {
    pub image_id: i64,
    pub scores: Vec<f32>,
    pub labels: Vec<i64>,
    /// Boxes in [x1, y1, x2, y2] format (xyxy)
    pub boxes: Vec<[f32; 4]>,
}

/// COCO annotation format for ground truth.
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[derive(Debug, Clone)]
pub struct CocoAnnotation {
    pub id: i64,
    pub image_id: i64,
    pub category_id: i64,
    /// Bbox in [x, y, width, height] format (xywh)
    pub bbox: [f32; 4],
    pub area: f32,
    pub iscrowd: i32,
}

/// COCO image info.
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[derive(Debug, Clone)]
pub struct CocoImage {
    pub id: i64,
    pub width: u32,
    pub height: u32,
    pub file_name: String,
}

/// COCO category info.
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[derive(Debug, Clone)]
pub struct CocoCategory {
    pub id: i64,
    pub name: String,
    pub supercategory: Option<String>,
}

/// COCO dataset (ground truth).
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[derive(Debug, Clone)]
pub struct CocoDataset {
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotation>,
    pub categories: Vec<CocoCategory>,
}

/// Ground truth box for evaluation.
#[derive(Debug, Clone)]
pub struct GroundTruthBox {
    pub bbox: [f32; 4], // xywh format
    pub category_id: i64,
    pub area: f32,
    pub iscrowd: bool,
}

/// Detection box for evaluation.
#[derive(Debug, Clone)]
pub struct DetectionBox {
    pub bbox: [f32; 4], // xywh format
    pub category_id: i64,
    pub score: f32,
}

/// COCO evaluation metrics.
#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Debug, Clone, Default)]
pub struct CocoMetrics {
    /// AP @ IoU=0.50:0.95
    pub ap: f32,
    /// AP @ IoU=0.50
    pub ap50: f32,
    /// AP @ IoU=0.75
    pub ap75: f32,
    /// AP for small objects (area < 32^2)
    pub ap_small: f32,
    /// AP for medium objects (32^2 < area < 96^2)
    pub ap_medium: f32,
    /// AP for large objects (area > 96^2)
    pub ap_large: f32,
    /// AR with max 1 detection per image
    pub ar_1: f32,
    /// AR with max 10 detections per image
    pub ar_10: f32,
    /// AR with max 100 detections per image
    pub ar_100: f32,
    /// AR for small objects
    pub ar_small: f32,
    /// AR for medium objects
    pub ar_medium: f32,
    /// AR for large objects
    pub ar_large: f32,
}

/// Area ranges for COCO evaluation.
#[derive(Debug, Clone, Copy)]
pub enum AreaRange {
    All,
    Small,  // area < 32^2 = 1024
    Medium, // 32^2 <= area < 96^2 = 9216
    Large,  // area >= 96^2
}

impl AreaRange {
    fn range(&self) -> (f32, f32) {
        match self {
            AreaRange::All => (0.0, 1e10),
            AreaRange::Small => (0.0, 1024.0),
            AreaRange::Medium => (1024.0, 9216.0),
            AreaRange::Large => (9216.0, 1e10),
        }
    }
}

/// Convert box from xyxy format to xywh format.
fn xyxy_to_xywh(xyxy: [f32; 4]) -> [f32; 4] {
    let [x1, y1, x2, y2] = xyxy;
    [x1, y1, x2 - x1, y2 - y1]
}

/// Compute IoU between two boxes in xywh format.
fn compute_iou(box1: [f32; 4], box2: [f32; 4]) -> f32 {
    let [x1, y1, w1, h1] = box1;
    let [x2, y2, w2, h2] = box2;

    let x1_min = x1;
    let y1_min = y1;
    let x1_max = x1 + w1;
    let y1_max = y1 + h1;

    let x2_min = x2;
    let y2_min = y2;
    let x2_max = x2 + w2;
    let y2_max = y2 + h2;

    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    let inter_w = (inter_x_max - inter_x_min).max(0.0);
    let inter_h = (inter_y_max - inter_y_min).max(0.0);
    let inter_area = inter_w * inter_h;

    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union_area = area1 + area2 - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Evaluation result for a single category at a single IoU threshold.
#[derive(Debug, Clone)]
struct CategoryEval {
    /// True positives and false positives, sorted by score descending.
    /// Each entry is (score, is_tp).
    detections: Vec<(f32, bool)>,
    /// Number of ground truth boxes (excluding crowd).
    num_gt: usize,
}

/// Compute precision-recall curve and return AP using pycocotools method.
///
/// This implements the exact same algorithm as pycocotools:
/// 1. Compute precision and recall at each detection
/// 2. Make precision monotonically decreasing (backward pass)
/// 3. Use searchsorted to find precision at 101 recall thresholds
fn compute_ap(detections: &[(f32, bool)], num_gt: usize) -> f32 {
    if num_gt == 0 {
        return 0.0;
    }

    if detections.is_empty() {
        return 0.0;
    }

    let mut tp_cumsum = 0;
    let mut fp_cumsum = 0;
    let mut precisions = Vec::with_capacity(detections.len());
    let mut recalls = Vec::with_capacity(detections.len());

    for &(_score, is_tp) in detections {
        if is_tp {
            tp_cumsum += 1;
        } else {
            fp_cumsum += 1;
        }
        // Add small epsilon to avoid division by zero (matches np.spacing(1))
        let precision = tp_cumsum as f32 / (tp_cumsum + fp_cumsum) as f32;
        let recall = tp_cumsum as f32 / num_gt as f32;
        precisions.push(precision);
        recalls.push(recall);
    }

    // Make precision monotonically decreasing (pycocotools backward pass)
    // pr[i-1] = max(pr[i-1], pr[i])
    for i in (1..precisions.len()).rev() {
        if precisions[i] > precisions[i - 1] {
            precisions[i - 1] = precisions[i];
        }
    }

    // COCO-style AP: 101-point interpolation using searchsorted
    let recall_thresholds: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
    let mut ap = 0.0;

    for &r_thresh in &recall_thresholds {
        // Find first index where recall >= r_thresh (searchsorted left)
        let idx = recalls.partition_point(|&r| r < r_thresh);
        if idx < precisions.len() {
            ap += precisions[idx];
        }
        // If no recall >= r_thresh, add 0 (implicit)
    }

    ap / recall_thresholds.len() as f32
}

/// Compute average recall at a given max detections limit.
fn compute_ar(detections: &[(f32, bool)], num_gt: usize, _max_dets: usize) -> f32 {
    if num_gt == 0 {
        return 0.0;
    }

    let tp_count: usize = detections.iter().filter(|(_, is_tp)| *is_tp).count();
    tp_count as f32 / num_gt as f32
}

/// COCO Evaluator.
pub struct CocoEvaluator {
    /// Ground truth annotations indexed by image_id.
    gt_by_image: HashMap<i64, Vec<GroundTruthBox>>,
    /// Set of valid image IDs.
    image_ids: Vec<i64>,
    /// Category IDs.
    category_ids: Vec<i64>,
    /// IoU thresholds for evaluation.
    iou_thresholds: Vec<f32>,
}

impl CocoEvaluator {
    /// Create a new evaluator from COCO ground truth.
    pub fn new(dataset: &CocoDataset) -> Self {
        // Index ground truth by image_id
        let mut gt_by_image: HashMap<i64, Vec<GroundTruthBox>> = HashMap::new();

        for ann in &dataset.annotations {
            let gt_box = GroundTruthBox {
                bbox: ann.bbox,
                category_id: ann.category_id,
                area: ann.area,
                iscrowd: ann.iscrowd != 0,
            };
            gt_by_image.entry(ann.image_id).or_default().push(gt_box);
        }

        let image_ids: Vec<i64> = dataset.images.iter().map(|img| img.id).collect();
        let category_ids: Vec<i64> = dataset.categories.iter().map(|cat| cat.id).collect();

        // Standard COCO IoU thresholds: 0.5:0.05:0.95
        let iou_thresholds: Vec<f32> = (0..10).map(|i| 0.5 + 0.05 * i as f32).collect();

        Self {
            gt_by_image,
            image_ids,
            category_ids,
            iou_thresholds,
        }
    }

    /// Evaluate predictions and return metrics.
    pub fn evaluate(&self, predictions: &HashMap<i64, ImagePrediction>) -> CocoMetrics {
        println!("\nRunning COCO evaluation...");
        println!("  {} images in ground truth", self.image_ids.len());
        println!("  {} images with predictions", predictions.len());
        println!("  {} categories", self.category_ids.len());
        println!(
            "  IoU thresholds: {:?}",
            self.iou_thresholds
                .iter()
                .map(|x| format!("{:.2}", x))
                .collect::<Vec<_>>()
        );

        // Compute metrics for different settings
        let ap = self.compute_ap_across_iou(predictions, AreaRange::All);
        let ap50 = self.compute_ap_at_iou(predictions, 0.5, AreaRange::All);
        let ap75 = self.compute_ap_at_iou(predictions, 0.75, AreaRange::All);
        let ap_small = self.compute_ap_across_iou(predictions, AreaRange::Small);
        let ap_medium = self.compute_ap_across_iou(predictions, AreaRange::Medium);
        let ap_large = self.compute_ap_across_iou(predictions, AreaRange::Large);

        let ar_1 = self.compute_ar(predictions, 1, AreaRange::All);
        let ar_10 = self.compute_ar(predictions, 10, AreaRange::All);
        let ar_100 = self.compute_ar(predictions, 100, AreaRange::All);
        let ar_small = self.compute_ar(predictions, 100, AreaRange::Small);
        let ar_medium = self.compute_ar(predictions, 100, AreaRange::Medium);
        let ar_large = self.compute_ar(predictions, 100, AreaRange::Large);

        CocoMetrics {
            ap,
            ap50,
            ap75,
            ap_small,
            ap_medium,
            ap_large,
            ar_1,
            ar_10,
            ar_100,
            ar_small,
            ar_medium,
            ar_large,
        }
    }

    /// Compute AP averaged across IoU thresholds.
    fn compute_ap_across_iou(
        &self,
        predictions: &HashMap<i64, ImagePrediction>,
        area_range: AreaRange,
    ) -> f32 {
        let mut total_ap = 0.0;
        for &iou_thresh in &self.iou_thresholds {
            total_ap += self.compute_ap_at_iou(predictions, iou_thresh, area_range);
        }
        total_ap / self.iou_thresholds.len() as f32
    }

    /// Compute AP at a specific IoU threshold.
    fn compute_ap_at_iou(
        &self,
        predictions: &HashMap<i64, ImagePrediction>,
        iou_threshold: f32,
        area_range: AreaRange,
    ) -> f32 {
        // Evaluate per category and average
        let mut category_aps = Vec::new();

        for &cat_id in &self.category_ids {
            let eval = self.evaluate_category(predictions, cat_id, iou_threshold, area_range, 100);
            if eval.num_gt > 0 {
                let ap = compute_ap(&eval.detections, eval.num_gt);
                category_aps.push(ap);
            }
        }

        if category_aps.is_empty() {
            0.0
        } else {
            category_aps.iter().sum::<f32>() / category_aps.len() as f32
        }
    }

    /// Compute AR at max detections limit.
    fn compute_ar(
        &self,
        predictions: &HashMap<i64, ImagePrediction>,
        max_dets: usize,
        area_range: AreaRange,
    ) -> f32 {
        // Average AR across IoU thresholds and categories
        let mut total_ar = 0.0;
        let mut count = 0;

        for &iou_thresh in &self.iou_thresholds {
            for &cat_id in &self.category_ids {
                let eval =
                    self.evaluate_category(predictions, cat_id, iou_thresh, area_range, max_dets);
                if eval.num_gt > 0 {
                    let ar = compute_ar(&eval.detections, eval.num_gt, max_dets);
                    total_ar += ar;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total_ar / count as f32
        }
    }

    /// Evaluate a single category following pycocotools logic.
    ///
    /// Key insight: area filtering is done on ground truth, not detections.
    /// GTs outside the area range are marked as "ignore".
    /// Detections matching ignored GTs are also ignored.
    /// Unmatched detections outside the area range are ignored (not counted as FP).
    fn evaluate_category(
        &self,
        predictions: &HashMap<i64, ImagePrediction>,
        category_id: i64,
        iou_threshold: f32,
        area_range: AreaRange,
        max_dets: usize,
    ) -> CategoryEval {
        let mut all_detections: Vec<(f32, bool)> = Vec::new(); // (score, is_tp)
        let mut total_gt = 0;

        let (area_min, area_max) = area_range.range();

        // Process each image
        for &image_id in &self.image_ids {
            // Get all ground truth for this image and category
            let all_gt: Vec<&GroundTruthBox> = self
                .gt_by_image
                .get(&image_id)
                .map(|boxes| {
                    boxes
                        .iter()
                        .filter(|b| b.category_id == category_id)
                        .collect()
                })
                .unwrap_or_default();

            // Determine which GTs are ignored (crowd or outside area range)
            let gt_ignore: Vec<bool> = all_gt
                .iter()
                .map(|g| g.iscrowd || g.area < area_min || g.area >= area_max)
                .collect();

            // Count non-ignored GTs
            let num_gt_this_image = gt_ignore.iter().filter(|&&ig| !ig).count();
            total_gt += num_gt_this_image;

            // Get detections for this image and category
            let mut det_boxes: Vec<DetectionBox> = predictions
                .get(&image_id)
                .map(|pred| {
                    pred.scores
                        .iter()
                        .zip(pred.labels.iter())
                        .zip(pred.boxes.iter())
                        .filter(|&((_, &label), _)| label == category_id)
                        .map(|((&score, _), &bbox)| DetectionBox {
                            bbox: xyxy_to_xywh(bbox),
                            category_id,
                            score,
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Sort by score descending and limit to max_dets
            det_boxes.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            det_boxes.truncate(max_dets);

            // Sort GTs: non-ignored first, then ignored
            let mut gt_order: Vec<usize> = (0..all_gt.len()).collect();
            gt_order.sort_by_key(|&i| gt_ignore[i] as u8);

            // Match detections to ground truth
            let mut gt_matched = vec![false; all_gt.len()];

            for det in &det_boxes {
                let det_area = det.bbox[2] * det.bbox[3];
                let mut best_iou = iou_threshold;
                let mut best_gt_idx: Option<usize> = None;

                // Find best matching ground truth (following sorted order)
                for &gt_idx in &gt_order {
                    let gt = all_gt[gt_idx];

                    // Skip if already matched (unless crowd)
                    if gt_matched[gt_idx] && !gt.iscrowd {
                        continue;
                    }

                    // If we found a match with non-ignored GT and now looking at ignored GT, stop
                    if best_gt_idx.is_some()
                        && !gt_ignore[best_gt_idx.unwrap()]
                        && gt_ignore[gt_idx]
                    {
                        break;
                    }

                    let iou = compute_iou(det.bbox, gt.bbox);
                    if iou > best_iou {
                        best_iou = iou;
                        best_gt_idx = Some(gt_idx);
                    }
                }

                // Determine if this detection should be ignored or counted
                if let Some(gt_idx) = best_gt_idx {
                    if gt_ignore[gt_idx] {
                        // Matched an ignored GT - don't count this detection
                        continue;
                    }
                    // True positive - matched a non-ignored GT
                    gt_matched[gt_idx] = true;
                    all_detections.push((det.score, true));
                } else {
                    // No match - check if detection should be ignored due to area
                    if det_area < area_min || det_area >= area_max {
                        // Detection outside area range and unmatched - ignore
                        continue;
                    }
                    // False positive
                    all_detections.push((det.score, false));
                }
            }
        }

        // Sort all detections by score descending
        all_detections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        CategoryEval {
            detections: all_detections,
            num_gt: total_gt,
        }
    }

    /// Print a summary of the metrics (COCO-style).
    pub fn print_summary(metrics: &CocoMetrics) {
        println!();
        println!(
            " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3}",
            metrics.ap
        );
        println!(
            " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3}",
            metrics.ap50
        );
        println!(
            " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3}",
            metrics.ap75
        );
        println!(
            " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3}",
            metrics.ap_small
        );
        println!(
            " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3}",
            metrics.ap_medium
        );
        println!(
            " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3}",
            metrics.ap_large
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3}",
            metrics.ar_1
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3}",
            metrics.ar_10
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3}",
            metrics.ar_100
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3}",
            metrics.ar_small
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3}",
            metrics.ar_medium
        );
        println!(
            " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3}",
            metrics.ar_large
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xyxy_to_xywh() {
        let xyxy = [10.0, 20.0, 50.0, 80.0];
        let xywh = xyxy_to_xywh(xyxy);
        assert_eq!(xywh, [10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_compute_iou() {
        // Perfect overlap
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let iou = compute_iou(box1, box1);
        assert!((iou - 1.0).abs() < 1e-6);

        // No overlap
        let box2 = [20.0, 20.0, 10.0, 10.0];
        let iou = compute_iou(box1, box2);
        assert!(iou.abs() < 1e-6);

        // 50% overlap
        let box3 = [5.0, 0.0, 10.0, 10.0];
        let iou = compute_iou(box1, box3);
        // Intersection: 5x10=50, Union: 100+100-50=150, IoU=50/150=0.333
        assert!((iou - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ap_perfect() {
        // All true positives
        let detections = vec![(0.9, true), (0.8, true), (0.7, true)];
        let ap = compute_ap(&detections, 3);
        assert!((ap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ap_half() {
        // Half true positives
        let detections = vec![(0.9, true), (0.8, false), (0.7, true), (0.6, false)];
        let ap = compute_ap(&detections, 2);
        // At recall 0.5: precision 1.0
        // At recall 1.0: precision 0.5
        assert!(ap > 0.0 && ap < 1.0);
    }
}
