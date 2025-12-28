import argparse
import time

import supervision as sv
from PIL import Image

from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)
from rfdetr.util.coco_classes import COCO_CLASSES

MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
    "seg": RFDETRSegPreview,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export RF-DETR model to ONNX format")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        choices=list(MODEL_CLASSES.keys()),
        help=f"Model size to export. Available: {', '.join(MODEL_CLASSES.keys())} (default: small)",
    )
    return parser.parse_args()


args = parse_args()

start_time = time.time()
model_cls = MODEL_CLASSES[args.model]
model = model_cls()
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.4f} seconds")

start_time = time.time()
model.optimize_for_inference()
optimize_time = time.time() - start_time
print(f"Model optimization time: {optimize_time:.4f} seconds")

# url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
# image = Image.open(io.BytesIO(requests.get(url).content))
image = Image.open("sample.jpg")
start_time = time.time()

detections = model.predict(image, threshold=0.5)

predict_time = time.time() - start_time
print(f"Model prediction time (without warmup): {predict_time:.4f} seconds")

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]
print(f"{len(labels)} objects detected:")
for label in labels:
    print(label)

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# save image to output.jpg
annotated_image.save("sample.py.jpg")

# sv.plot_image(annotated_image)
