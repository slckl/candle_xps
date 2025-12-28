import argparse
import os
import shutil
import time

from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)

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
output_dir = "./output"
start_time = time.time()
model_cls = MODEL_CLASSES[args.model]
model = model_cls()
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
