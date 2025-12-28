"""
Export RF-DETR model weights to safetensors format for use with Candle.

This script loads a pre-trained RF-DETR model and exports its weights
to the safetensors format, which can be loaded by the Rust implementation.

Usage:
    uv run export_safetensors.py --model medium --output rf-detr-medium.safetensors
"""

import argparse
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file

from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
)

MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def flatten_state_dict(state_dict: dict, prefix: str = "") -> OrderedDict:
    """
    Flatten a nested state dict into a single-level dict with dot-separated keys.
    """
    result = OrderedDict()
    for key, value in state_dict.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_state_dict(value, f"{new_key}."))
        elif isinstance(value, torch.Tensor):
            result[new_key] = value.contiguous()
        else:
            # Skip non-tensor values
            pass
    return result


def rename_keys_for_candle(state_dict: OrderedDict) -> OrderedDict:
    """
    Rename keys to match the expected naming convention in the Candle implementation.

    The RF-DETR PyTorch model has this structure:
    - backbone.0.encoder.encoder.embeddings.* (patch embeddings, position embeddings, cls token)
    - backbone.0.encoder.encoder.encoder.layer.* (transformer blocks)
    - backbone.0.projector.* (multi-scale projector)
    - transformer.decoder.* (DETR decoder)
    - transformer.enc_output.* (two-stage encoder outputs)
    - class_embed.* (classification head)
    - bbox_embed.* (bounding box head)
    - refpoint_embed.* (reference point embeddings)
    - query_feat.* (query features)

    We map these to a simplified structure for Candle.
    """
    renamed = OrderedDict()

    for key, value in state_dict.items():
        new_key = key

        # =====================================================================
        # Backbone encoder (DINOv2) mappings
        # =====================================================================

        # Map: backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.*
        # To:  backbone.0.encoder.embeddings.patch_embeddings.projection.*
        if "backbone.0.encoder.encoder.embeddings." in new_key:
            new_key = new_key.replace(
                "backbone.0.encoder.encoder.embeddings.",
                "backbone.0.encoder.embeddings.",
            )

        # Map: backbone.0.encoder.encoder.encoder.layer.X.*
        # To:  backbone.0.encoder.encoder.layer.X.*
        if "backbone.0.encoder.encoder.encoder.layer." in new_key:
            new_key = new_key.replace(
                "backbone.0.encoder.encoder.encoder.layer.",
                "backbone.0.encoder.encoder.layer.",
            )

        # Map attention structure:
        # From: .attention.attention.query/key/value.*
        # To:   .attention.q_proj/k_proj/v_proj.*
        if ".attention.attention.query." in new_key:
            new_key = new_key.replace(
                ".attention.attention.query.", ".attention.q_proj."
            )
        if ".attention.attention.key." in new_key:
            new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
        if ".attention.attention.value." in new_key:
            new_key = new_key.replace(
                ".attention.attention.value.", ".attention.v_proj."
            )
        if ".attention.output.dense." in new_key:
            new_key = new_key.replace(
                ".attention.output.dense.", ".attention.out_proj."
            )

        # =====================================================================
        # Transformer decoder self-attention mappings
        # =====================================================================

        # PyTorch nn.MultiheadAttention uses in_proj_weight/in_proj_bias (combined QKV)
        # We need to split these for Candle's separate Q, K, V projections
        # This is handled separately in split_combined_qkv below

        renamed[new_key] = value

    return renamed


def split_combined_qkv(state_dict: OrderedDict, hidden_dim: int = 256) -> OrderedDict:
    """
    Split combined QKV projections from nn.MultiheadAttention into separate Q, K, V.

    PyTorch's nn.MultiheadAttention uses:
    - in_proj_weight: [3*hidden_dim, hidden_dim] -> combined Q, K, V weights
    - in_proj_bias: [3*hidden_dim] -> combined Q, K, V biases

    We split these into:
    - q_proj.weight, k_proj.weight, v_proj.weight
    - q_proj.bias, k_proj.bias, v_proj.bias
    """
    result = OrderedDict()
    keys_to_remove = set()

    for key, value in state_dict.items():
        if ".self_attn.in_proj_weight" in key:
            prefix = key.replace(".in_proj_weight", "")
            q_weight, k_weight, v_weight = value.chunk(3, dim=0)
            result[f"{prefix}.q_proj.weight"] = q_weight
            result[f"{prefix}.k_proj.weight"] = k_weight
            result[f"{prefix}.v_proj.weight"] = v_weight
            keys_to_remove.add(key)

        elif ".self_attn.in_proj_bias" in key:
            prefix = key.replace(".in_proj_bias", "")
            q_bias, k_bias, v_bias = value.chunk(3, dim=0)
            result[f"{prefix}.q_proj.bias"] = q_bias
            result[f"{prefix}.k_proj.bias"] = k_bias
            result[f"{prefix}.v_proj.bias"] = v_bias
            keys_to_remove.add(key)

    # Copy all non-split keys
    for key, value in state_dict.items():
        if key not in keys_to_remove:
            result[key] = value

    return result


def export_model(model_name: str, output_path: str, optimize: bool = True) -> None:
    """
    Export the specified RF-DETR model to safetensors format.

    Args:
        model_name: Name of the model variant (nano, small, medium, base, large)
        output_path: Path to save the safetensors file
        optimize: Whether to optimize the model for inference before export
    """
    print(f"Loading RF-DETR {model_name} model...")

    model_class = MODEL_CLASSES.get(model_name.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CLASSES.keys())}"
        )

    # Initialize model (this downloads weights if not present)
    model = model_class()

    # Get the underlying PyTorch model
    pytorch_model = model.model.model
    pytorch_model.eval()

    # Export mode preparation
    pytorch_model.export()

    # Get state dict
    state_dict = pytorch_model.state_dict()

    print(f"Model has {len(state_dict)} parameters")

    # Print some sample keys for debugging
    print("\nSample parameter keys (before renaming):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}: {state_dict[key].shape}")
    print("  ...")

    # Flatten nested dicts if any
    flat_state_dict = flatten_state_dict(state_dict)

    # Rename keys for Candle compatibility
    renamed_state_dict = rename_keys_for_candle(flat_state_dict)

    # Split combined QKV projections
    config = model.model_config
    hidden_dim = config.hidden_dim
    split_state_dict = split_combined_qkv(renamed_state_dict, hidden_dim)

    # Ensure all tensors are contiguous and on CPU
    final_state_dict = OrderedDict()
    for key, value in split_state_dict.items():
        if isinstance(value, torch.Tensor):
            final_state_dict[key] = value.cpu().contiguous()

    print(f"\nSample parameter keys (after renaming):")
    for i, key in enumerate(list(final_state_dict.keys())[:10]):
        print(f"  {key}: {final_state_dict[key].shape}")
    print("  ...")

    print(f"\nExporting {len(final_state_dict)} tensors to {output_path}")

    # Save to safetensors format
    save_file(final_state_dict, output_path)

    # Verify the export
    file_size = os.path.getsize(output_path)
    print(
        f"Successfully exported model to {output_path} ({file_size / 1024 / 1024:.2f} MB)"
    )

    # Print model configuration for reference
    print(f"\nModel configuration:")
    print(f"  Resolution: {config.resolution}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Decoder layers: {config.dec_layers}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Num queries: {getattr(config, 'num_queries', 300)}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Encoder: {config.encoder}")
    print(f"  Out feature indexes: {config.out_feature_indexes}")


def print_model_keys(model_name: str) -> None:
    """
    Print all keys in the model's state dict (useful for debugging).
    """
    print(f"Loading RF-DETR {model_name} model...")

    model_class = MODEL_CLASSES.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_class()
    pytorch_model = model.model.model
    pytorch_model.eval()

    state_dict = pytorch_model.state_dict()

    print(f"\nAll {len(state_dict)} parameter keys:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape} ({state_dict[key].dtype})")


def print_renamed_keys(model_name: str) -> None:
    """
    Print all keys after renaming (useful for debugging).
    """
    print(f"Loading RF-DETR {model_name} model...")

    model_class = MODEL_CLASSES.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_class()
    pytorch_model = model.model.model
    pytorch_model.eval()
    pytorch_model.export()

    state_dict = pytorch_model.state_dict()
    flat_state_dict = flatten_state_dict(state_dict)
    renamed_state_dict = rename_keys_for_candle(flat_state_dict)

    config = model.model_config
    split_state_dict = split_combined_qkv(renamed_state_dict, config.hidden_dim)

    print(f"\nAll {len(split_state_dict)} parameter keys (after renaming):")
    for key in sorted(split_state_dict.keys()):
        print(f"  {key}: {split_state_dict[key].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export RF-DETR model weights to safetensors format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=list(MODEL_CLASSES.keys()),
        help="Model variant to export (default: medium)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for safetensors file (default: rf-detr-{model}.safetensors)",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List all model keys instead of exporting",
    )
    parser.add_argument(
        "--list-renamed-keys",
        action="store_true",
        help="List all model keys after renaming",
    )

    args = parser.parse_args()

    if args.list_keys:
        print_model_keys(args.model)
        return

    if args.list_renamed_keys:
        print_renamed_keys(args.model)
        return

    output_path = args.output
    if output_path is None:
        output_path = f"rf-detr-{args.model}.safetensors"

    export_model(args.model, output_path)


if __name__ == "__main__":
    main()
