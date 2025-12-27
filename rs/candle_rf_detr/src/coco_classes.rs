/// COCO class names for RF-DETR object detection.
/// RF-DETR uses 1-indexed class IDs (1-90), matching the original COCO dataset.
/// Note: Some IDs are skipped in the original COCO dataset (e.g., 12, 26, 29, 30, etc.)

pub const NAMES: [&str; 91] = [
    "background",    // 0 - background/no object
    "person",        // 1
    "bicycle",       // 2
    "car",           // 3
    "motorcycle",    // 4
    "airplane",      // 5
    "bus",           // 6
    "train",         // 7
    "truck",         // 8
    "boat",          // 9
    "traffic light", // 10
    "fire hydrant",  // 11
    "N/A",           // 12 - not used in COCO
    "stop sign",     // 13
    "parking meter", // 14
    "bench",         // 15
    "bird",          // 16
    "cat",           // 17
    "dog",           // 18
    "horse",         // 19
    "sheep",         // 20
    "cow",           // 21
    "elephant",      // 22
    "bear",          // 23
    "zebra",         // 24
    "giraffe",       // 25
    "N/A",           // 26 - not used in COCO
    "backpack",      // 27
    "umbrella",      // 28
    "N/A",           // 29 - not used in COCO
    "N/A",           // 30 - not used in COCO
    "handbag",       // 31
    "tie",           // 32
    "suitcase",      // 33
    "frisbee",       // 34
    "skis",          // 35
    "snowboard",     // 36
    "sports ball",   // 37
    "kite",          // 38
    "baseball bat",  // 39
    "baseball glove", // 40
    "skateboard",    // 41
    "surfboard",     // 42
    "tennis racket", // 43
    "bottle",        // 44
    "N/A",           // 45 - not used in COCO
    "wine glass",    // 46
    "cup",           // 47
    "fork",          // 48
    "knife",         // 49
    "spoon",         // 50
    "bowl",          // 51
    "banana",        // 52
    "apple",         // 53
    "sandwich",      // 54
    "orange",        // 55
    "broccoli",      // 56
    "carrot",        // 57
    "hot dog",       // 58
    "pizza",         // 59
    "donut",         // 60
    "cake",          // 61
    "chair",         // 62
    "couch",         // 63
    "potted plant",  // 64
    "bed",           // 65
    "N/A",           // 66 - not used in COCO
    "dining table",  // 67
    "N/A",           // 68 - not used in COCO
    "N/A",           // 69 - not used in COCO
    "toilet",        // 70
    "N/A",           // 71 - not used in COCO
    "tv",            // 72
    "laptop",        // 73
    "mouse",         // 74
    "remote",        // 75
    "keyboard",      // 76
    "cell phone",    // 77
    "microwave",     // 78
    "oven",          // 79
    "toaster",       // 80
    "sink",          // 81
    "refrigerator",  // 82
    "N/A",           // 83 - not used in COCO
    "book",          // 84
    "clock",         // 85
    "vase",          // 86
    "scissors",      // 87
    "teddy bear",    // 88
    "hair drier",    // 89
    "toothbrush",    // 90
];

/// Get class name by ID, returns "unknown" for invalid IDs
pub fn get_class_name(class_id: usize) -> &'static str {
    if class_id < NAMES.len() {
        NAMES[class_id]
    } else {
        "unknown"
    }
}
