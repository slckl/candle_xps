import io
import os
import shutil
import time

import requests
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

# This also downloads model, if it's not present in working directory of script.

output_dir = "./output"
start_time = time.time()
# model = RFDETRMedium()
model = RFDETRSmall()
model_name = model.size
print(f"Loaded model: {model_name}")
model_load_time = time.time() - start_time
print(f"Model loading time: {model_load_time:.4f} seconds")

start_time = time.time()
# simplification does not seem to be working rn for some reason
simplify = False
model.export(
    output_dir=output_dir,
    simplify=simplify,
)
print(f"Model export time: {time.time() - start_time:.4f} seconds")

# Rename exported inference_model.onnx to {model_name}{.full/.simplified}.onnx
src_file = "inference_model.onnx"
if simplify:
    src_file = "inference_model.sim.onnx"
suffix = ".simplified.onnx" if simplify else ".full.onnx"
new_model_path = os.path.join(output_dir, f"{model_name}{suffix}")
shutil.move(
    os.path.join(output_dir, src_file),
    new_model_path,
)
print(f"Exported model saved to: {new_model_path}")

# start_time = time.time()
# model.optimize_for_inference()
# optimize_time = time.time() - start_time
# print(f"Model optimization time: {optimize_time:.4f} seconds")

# # url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
# # image = Image.open(io.BytesIO(requests.get(url).content))
# image = Image.open("sample.jpg")
# warmup = 10
# iters = 10
# start_time = None
# for i in range(warmup + iters):
#     if i > warmup and start_time is None:
#         start_time = time.time()
#     detections = model.predict(image, threshold=0.5)

# predict_time = (time.time() - start_time) / iters
# print(f"Avg model prediction time: {predict_time:.4f} seconds")

# labels = [
#     f"{COCO_CLASSES[class_id]} {confidence:.2f}"
#     for class_id, confidence in zip(detections.class_id, detections.confidence)
# ]

# annotated_image = image.copy()
# annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# # save image to output.jpg
# annotated_image.save("output.jpg")

# sv.plot_image(annotated_image)
