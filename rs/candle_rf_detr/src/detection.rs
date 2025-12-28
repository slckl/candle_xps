//! Detection output structure
//!
//! This module defines the Detection struct that represents a single object detection result.

use std::fmt;

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
}

impl Detection {
    /// Create a new detection result.
    ///
    /// # Arguments
    /// * `bbox` - Bounding box as [x1, y1, x2, y2]
    /// * `score` - Confidence score
    /// * `class_id` - Class ID
    pub fn new(bbox: [f32; 4], score: f32, class_id: usize) -> Self {
        Self {
            bbox,
            score,
            class_id,
        }
    }

    /// Get the width of the bounding box
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    /// Get the height of the bounding box
    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> (f32, f32) {
        let cx = (self.bbox[0] + self.bbox[2]) / 2.0;
        let cy = (self.bbox[1] + self.bbox[3]) / 2.0;
        (cx, cy)
    }

    /// Convert bounding box from [x1, y1, x2, y2] to [cx, cy, w, h] format
    pub fn to_cxcywh(&self) -> [f32; 4] {
        let (cx, cy) = self.center();
        [cx, cy, self.width(), self.height()]
    }

    /// Create a detection from [cx, cy, w, h] format bounding box
    pub fn from_cxcywh(cxcywh: [f32; 4], score: f32, class_id: usize) -> Self {
        let [cx, cy, w, h] = cxcywh;
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;
        Self::new([x1, y1, x2, y2], score, class_id)
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Detection {{ class_id: {}, score: {:.2}, bbox: [{:.1}, {:.1}, {:.1}, {:.1}] }}",
            self.class_id, self.score, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_dimensions() {
        let det = Detection::new([10.0, 20.0, 110.0, 170.0], 0.9, 1);
        assert_eq!(det.width(), 100.0);
        assert_eq!(det.height(), 150.0);
        assert_eq!(det.area(), 15000.0);
    }

    #[test]
    fn test_detection_center() {
        let det = Detection::new([0.0, 0.0, 100.0, 100.0], 0.9, 1);
        assert_eq!(det.center(), (50.0, 50.0));
    }

    #[test]
    fn test_cxcywh_conversion() {
        let det = Detection::new([10.0, 20.0, 110.0, 120.0], 0.9, 1);
        let cxcywh = det.to_cxcywh();
        assert_eq!(cxcywh, [60.0, 70.0, 100.0, 100.0]);

        let det2 = Detection::from_cxcywh(cxcywh, 0.9, 1);
        assert_eq!(det2.bbox, det.bbox);
    }
}
