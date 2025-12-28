//! COCO Classes
//!
//! This module provides the mapping from COCO class IDs to class names.
//! Note: COCO class IDs are not contiguous (some IDs are skipped).

/// COCO class names indexed by class ID.
/// The model outputs class indices 0-90, where index 0 is typically background
/// and indices 1-90 map to COCO category IDs.
///
/// Note: Some indices have no corresponding class (marked as empty strings).
/// These correspond to gaps in the original COCO category IDs.
pub const COCO_CLASSES: [&str; 91] = [
    "background",     // 0 - background/no object
    "person",         // 1
    "bicycle",        // 2
    "car",            // 3
    "motorcycle",     // 4
    "airplane",       // 5
    "bus",            // 6
    "train",          // 7
    "truck",          // 8
    "boat",           // 9
    "traffic light",  // 10
    "fire hydrant",   // 11
    "",               // 12 - not used in COCO
    "stop sign",      // 13
    "parking meter",  // 14
    "bench",          // 15
    "bird",           // 16
    "cat",            // 17
    "dog",            // 18
    "horse",          // 19
    "sheep",          // 20
    "cow",            // 21
    "elephant",       // 22
    "bear",           // 23
    "zebra",          // 24
    "giraffe",        // 25
    "",               // 26 - not used in COCO
    "backpack",       // 27
    "umbrella",       // 28
    "",               // 29 - not used in COCO
    "",               // 30 - not used in COCO
    "handbag",        // 31
    "tie",            // 32
    "suitcase",       // 33
    "frisbee",        // 34
    "skis",           // 35
    "snowboard",      // 36
    "sports ball",    // 37
    "kite",           // 38
    "baseball bat",   // 39
    "baseball glove", // 40
    "skateboard",     // 41
    "surfboard",      // 42
    "tennis racket",  // 43
    "bottle",         // 44
    "",               // 45 - not used in COCO
    "wine glass",     // 46
    "cup",            // 47
    "fork",           // 48
    "knife",          // 49
    "spoon",          // 50
    "bowl",           // 51
    "banana",         // 52
    "apple",          // 53
    "sandwich",       // 54
    "orange",         // 55
    "broccoli",       // 56
    "carrot",         // 57
    "hot dog",        // 58
    "pizza",          // 59
    "donut",          // 60
    "cake",           // 61
    "chair",          // 62
    "couch",          // 63
    "potted plant",   // 64
    "bed",            // 65
    "",               // 66 - not used in COCO
    "dining table",   // 67
    "",               // 68 - not used in COCO
    "",               // 69 - not used in COCO
    "toilet",         // 70
    "",               // 71 - not used in COCO
    "tv",             // 72
    "laptop",         // 73
    "mouse",          // 74
    "remote",         // 75
    "keyboard",       // 76
    "cell phone",     // 77
    "microwave",      // 78
    "oven",           // 79
    "toaster",        // 80
    "sink",           // 81
    "refrigerator",   // 82
    "",               // 83 - not used in COCO
    "book",           // 84
    "clock",          // 85
    "vase",           // 86
    "scissors",       // 87
    "teddy bear",     // 88
    "hair drier",     // 89
    "toothbrush",     // 90
];

/// Get the class name for a given class ID.
///
/// # Arguments
/// * `class_id` - The class ID (0-90)
///
/// # Returns
/// The class name, or "unknown" if the ID is out of range or unused.
pub fn get_class_name(class_id: usize) -> &'static str {
    if class_id < COCO_CLASSES.len() {
        let name = COCO_CLASSES[class_id];
        if name.is_empty() {
            "unknown"
        } else {
            name
        }
    } else {
        "unknown"
    }
}

/// Get the number of COCO classes (including background)
pub const fn num_classes() -> usize {
    COCO_CLASSES.len()
}
