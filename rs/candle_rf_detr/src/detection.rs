/// Raw prediction output before thresholding.
/// Contains all queries with their scores, labels, and boxes.
#[derive(Debug, Clone)]
pub struct RawPrediction {
    /// Confidence scores for each query (max across classes)
    pub scores: Vec<f32>,
    /// Class labels for each query (argmax across classes)
    pub labels: Vec<i64>,
    /// Bounding boxes in [x1, y1, x2, y2] pixel coordinates
    pub boxes: Vec<[f32; 4]>,
}

/// Optional segmentation mask for a detection.
#[derive(Debug, Clone)]
pub struct Mask {
    /// Binary mask for this detection [H, W] as flattened Vec<bool>
    pub mask: Vec<bool>,
    /// Mask dimensions (height, width)
    pub mask_dims: (usize, usize),
}

/// A single object detection result.
///
/// Contains the bounding box coordinates, confidence score, and class information
/// for a detected object.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in [x1, y1, x2, y2] format (top-left and bottom-right corners)
    /// Coordinates are in pixel space of the original image.
    pub bbox: [f32; 4],
    /// Confidence score for this detection (0.0 to 1.0)
    pub score: f32,
    /// Class ID (0-90 for COCO, where 0 is background)
    pub class_id: usize,
    /// Optional segmentation mask for this detection.
    pub mask: Option<Mask>,
}
