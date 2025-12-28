"""
Export RF-DETR model weights to safetensors format.

This script loads a pre-trained RF-DETR model and exports its weights
to the safetensors format.

Usage:
    uv run export_safetensors.py --model small
    uv run export_safetensors.py --model medium --output ./output/custom-name.safetensors
"""

import argparse
import os
import time
from collections import OrderedDict

import torch
from safetensors.torch import save_file

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
    parser = argparse.ArgumentParser(
        description="Export RF-DETR model to safetensors format"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        choices=list(MODEL_CLASSES.keys()),
        help=f"Model size to export. Available: {', '.join(MODEL_CLASSES.keys())} (default: small)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for safetensors file (default: ./output/{model_name}.safetensors)",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List all model parameter keys and exit without exporting",
    )
    return parser.parse_args()


def prepare_state_dict(state_dict: dict) -> OrderedDict:
    """
    Prepare state dict for safetensors export.
    Ensures all tensors are contiguous and on CPU.
    """
    result = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Ensure tensor is contiguous and on CPU
            result[key] = value.cpu().contiguous()
    return result


def list_model_keys(model_name: str) -> None:
    """
    List all parameter keys in the model's state dict.
    """
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls()

    pytorch_model = model.model.model
    pytorch_model.eval()
    pytorch_model.export()

    state_dict = pytorch_model.state_dict()

    print(f"\nModel: {model.size}")
    print(f"Total parameters: {len(state_dict)}")
    print(f"\nAll parameter keys:")
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        print(f"  {key}: {list(value.shape)} ({value.dtype})")


def main():
    args = parse_args()

    # Handle --list-keys option
    if args.list_keys:
        list_model_keys(args.model)
        return

    output_dir = "./export"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    start_time = time.time()
    model_cls = MODEL_CLASSES[args.model]
    model = model_cls()
    model_name = model.size
    print(f"Loaded model: {model_name}")
    model_load_time = time.time() - start_time
    print(f"Model loading time: {model_load_time:.4f} seconds")

    # Get the underlying PyTorch model and set to eval/export mode
    pytorch_model = model.model.model
    pytorch_model.eval()
    pytorch_model.export()

    # Get state dict
    state_dict = pytorch_model.state_dict()
    print(f"Model has {len(state_dict)} parameters")

    # Print some sample keys for reference
    print("\nSample parameter keys:")
    for i, (key, value) in enumerate(list(state_dict.items())[:10]):
        print(f"  {key}: {list(value.shape)} ({value.dtype})")
    print("  ...")

    # Prepare state dict for export
    start_time = time.time()
    final_state_dict = prepare_state_dict(state_dict)

    # Determine output path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = os.path.join(output_dir, f"{model_name}.safetensors")

    # Save to safetensors format
    save_file(final_state_dict, output_path)
    export_time = time.time() - start_time
    print(f"\nModel export time: {export_time:.4f} seconds")

    # Report file size
    file_size = os.path.getsize(output_path)
    print(f"Exported model saved to: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

    # Print model configuration for reference
    config = model.model_config
    print(f"\nModel configuration:")
    print(f"  Resolution: {config.resolution}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Decoder layers: {config.dec_layers}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Num queries: {getattr(config, 'num_queries', 300)}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Encoder: {config.encoder}")
    print(f"  Out feature indexes: {config.out_feature_indexes}")


if __name__ == "__main__":
    main()
