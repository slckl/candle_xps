"""
RF-DETR COCO Evaluation Script

This script evaluates RF-DETR models on the COCO validation dataset and reports
standard object detection metrics (AP, AP50, AP75, etc.).

Features:
- Saves intermediate predictions to disk for later analysis
- Can load cached predictions to skip inference
- Reports standard COCO metrics

Usage:
    # Run evaluation and save predictions
    uv run eval_study.py -m nano --coco-path /path/to/coco

    # Load cached predictions and evaluate
    uv run eval_study.py -m nano --coco-path /path/to/coco --load-predictions

    # Specify custom output directory
    uv run eval_study.py -m nano --coco-path /path/to/coco -o my_eval_output
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import rfdetr.util.misc as utils
from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
)
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.engine import coco_extended_metrics
from rfdetr.models import build_criterion_and_postprocessors

MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

# Model configurations matching rfdetr defaults
MODEL_CONFIGS = {
    "nano": {
        "resolution": 384,
        "hidden_dim": 256,
        "dec_layers": 2,
        "encoder": "dinov2_windowed_small",
        "projector_scale": ["P4"],
    },
    "small": {
        "resolution": 512,
        "hidden_dim": 256,
        "dec_layers": 3,
        "encoder": "dinov2_windowed_small",
        "projector_scale": ["P4"],
    },
    "medium": {
        "resolution": 576,
        "hidden_dim": 256,
        "dec_layers": 4,
        "encoder": "dinov2_windowed_small",
        "projector_scale": ["P4"],
    },
    "base": {
        "resolution": 560,
        "hidden_dim": 256,
        "dec_layers": 5,
        "encoder": "dinov2_windowed_base",
        "projector_scale": ["P4"],
    },
    "large": {
        "resolution": 560,
        "hidden_dim": 256,
        "dec_layers": 6,
        "encoder": "dinov2_windowed_base",
        "projector_scale": ["P3", "P5"],
    },
}


class Args:
    """Arguments namespace for evaluation configuration."""

    def __init__(self, model_name: str, coco_path: str, **kwargs):
        config = MODEL_CONFIGS[model_name]

        # Dataset settings
        self.dataset_file = "coco"
        self.coco_path = coco_path
        self.resolution = config["resolution"]

        # Model architecture
        self.num_classes = 91  # COCO classes
        self.hidden_dim = config["hidden_dim"]
        self.dec_layers = config["dec_layers"]
        self.encoder = config["encoder"]
        self.num_queries = 300
        self.num_select = 300
        self.group_detr = 1  # inference mode
        self.two_stage = True
        self.lite_refpoint_refine = True
        self.bbox_reparam = True
        self.projector_scale = config["projector_scale"]
        self.dec_n_points = 4
        self.decoder_norm = "LN"
        self.sa_nheads = 8
        self.ca_nheads = 8
        self.dim_feedforward = 2048
        self.aux_loss = False
        self.segmentation_head = False

        # Loss settings (needed for criterion)
        self.cls_loss_coef = 2.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.set_cost_class = 2.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0
        self.focal_alpha = 0.25
        self.use_varifocal_loss = False
        self.use_position_supervised_loss = False
        self.ia_bce_loss = False
        self.sum_group_losses = False

        # Evaluation settings
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = kwargs.get("batch_size", 1)
        self.num_workers = kwargs.get("num_workers", 4)
        self.fp16_eval = kwargs.get("fp16_eval", False)

        # Dataset transform settings
        self.multi_scale = False
        self.expanded_scales = False
        self.do_random_resize_via_padding = False
        self.square_resize_div_64 = True  # Use square resize for proper model input
        self.patch_size = 16
        self.num_windows = 4

        # Other required args
        self.distributed = False
        self.output_dir = kwargs.get("output_dir", "eval_output")


def save_predictions(predictions: dict, output_dir: Path, model_name: str):
    """
    Save predictions to disk for later analysis.

    Args:
        predictions: Dict mapping image_id to prediction results
        output_dir: Directory to save predictions
        model_name: Name of the model (used in filename)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle for exact reproduction
    pkl_path = output_dir / f"predictions_{model_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Saved predictions to: {pkl_path}")

    # Also save a JSON summary for human inspection
    json_path = output_dir / f"predictions_{model_name}_summary.json"
    summary = {
        "num_images": len(predictions),
        "image_ids": sorted(predictions.keys()),
        "sample_predictions": {},
    }

    # Include a few sample predictions for inspection
    sample_ids = sorted(predictions.keys())[:5]
    for img_id in sample_ids:
        pred = predictions[img_id]
        summary["sample_predictions"][img_id] = {
            "num_detections": len(pred["scores"]),
            "scores": pred["scores"][:10].tolist() if len(pred["scores"]) > 0 else [],
            "labels": pred["labels"][:10].tolist() if len(pred["labels"]) > 0 else [],
            "boxes": pred["boxes"][:10].tolist() if len(pred["boxes"]) > 0 else [],
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved prediction summary to: {json_path}")


def load_predictions(output_dir: Path, model_name: str) -> dict:
    """
    Load predictions from disk.

    Args:
        output_dir: Directory containing saved predictions
        model_name: Name of the model

    Returns:
        Dict mapping image_id to prediction results
    """
    pkl_path = output_dir / f"predictions_{model_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pkl_path}\n"
            "Run evaluation without --load-predictions first to generate predictions."
        )

    print(f"Loading predictions from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        predictions = pickle.load(f)
    print(f"Loaded {len(predictions)} image predictions")
    return predictions


def run_inference(
    model, postprocess, data_loader, device, args, output_dir: Path, model_name: str
) -> dict:
    """
    Run model inference on the dataset and save predictions.

    Returns:
        Dict mapping image_id to prediction results
    """
    model.eval()

    if args.fp16_eval:
        model.half()

    all_predictions = {}
    num_batches = len(data_loader)
    print(f"Running inference on {num_batches} batches...")

    start_time = time.time()
    processed = 0

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        with torch.inference_mode():
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].float()

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocess(outputs, orig_target_sizes)

        # Store predictions for each image
        for target, result in zip(targets, results):
            img_id = target["image_id"].item()
            # Convert tensors to CPU numpy for storage
            all_predictions[img_id] = {
                "scores": result["scores"].cpu().numpy(),
                "labels": result["labels"].cpu().numpy(),
                "boxes": result["boxes"].cpu().numpy(),
            }

        processed += len(targets)
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            imgs_per_sec = processed / elapsed
            print(
                f"  [{batch_idx + 1}/{num_batches}] "
                f"Processed {processed} images ({imgs_per_sec:.1f} img/s)"
            )

    total_time = time.time() - start_time
    print(
        f"\nInference completed in {total_time:.1f}s ({processed / total_time:.1f} img/s)"
    )

    # Save predictions to disk
    save_predictions(all_predictions, output_dir, model_name)

    return all_predictions


def evaluate_predictions(predictions: dict, base_ds, iou_types=("bbox",)) -> dict:
    """
    Evaluate predictions using COCO evaluator.

    Args:
        predictions: Dict mapping image_id to prediction results
        base_ds: COCO API object for the dataset
        iou_types: Tuple of IoU types to evaluate

    Returns:
        Dict containing evaluation statistics
    """
    print("\nRunning COCO evaluation...")

    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # Convert predictions back to tensor format expected by evaluator
    tensor_predictions = {}
    for img_id, pred in predictions.items():
        tensor_predictions[img_id] = {
            "scores": torch.from_numpy(pred["scores"]),
            "labels": torch.from_numpy(pred["labels"]),
            "boxes": torch.from_numpy(pred["boxes"]),
        }

    # Update evaluator with all predictions
    print(f"Evaluating {len(tensor_predictions)} images...")
    coco_evaluator.update(tensor_predictions)

    # Synchronize and accumulate results
    print("Accumulating evaluation results...")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Get detailed metrics
    stats = {}
    if "bbox" in iou_types:
        bbox_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
        stats["bbox"] = {
            "AP": bbox_stats[0],
            "AP50": bbox_stats[1],
            "AP75": bbox_stats[2],
            "AP_small": bbox_stats[3],
            "AP_medium": bbox_stats[4],
            "AP_large": bbox_stats[5],
            "AR_1": bbox_stats[6],
            "AR_10": bbox_stats[7],
            "AR_100": bbox_stats[8],
            "AR_small": bbox_stats[9],
            "AR_medium": bbox_stats[10],
            "AR_large": bbox_stats[11],
        }
        # Get extended metrics if available
        try:
            extended = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
            stats["extended"] = extended
        except Exception:
            pass

    return stats, coco_evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="RF-DETR COCO Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        choices=list(MODEL_CLASSES.keys()),
        help=f"Model size to evaluate. Available: {', '.join(MODEL_CLASSES.keys())}",
    )
    parser.add_argument(
        "--coco-path",
        type=str,
        required=True,
        help="Path to COCO dataset root (should contain 'annotations' and 'val2017' folders)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="eval_output",
        help="Directory to save evaluation results and predictions",
    )
    parser.add_argument(
        "--load-predictions",
        "-l",
        action="store_true",
        help="Load cached predictions instead of running inference",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("RF-DETR COCO Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"COCO path: {args.coco_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"FP16: {args.fp16}")
    print(f"Output dir: {args.output_dir}")
    print(f"Load predictions: {args.load_predictions}")
    print("=" * 60)

    # Verify COCO path
    coco_path = Path(args.coco_path)
    if not coco_path.exists():
        raise ValueError(f"COCO path does not exist: {coco_path}")

    ann_file = coco_path / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        raise ValueError(
            f"COCO annotations not found: {ann_file}\n"
            "Expected directory structure:\n"
            "  {coco_path}/\n"
            "    annotations/\n"
            "      instances_val2017.json\n"
            "    val2017/\n"
            "      *.jpg"
        )

    # Create evaluation args
    eval_args = Args(
        args.model,
        args.coco_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fp16_eval=args.fp16,
        output_dir=args.output_dir,
    )

    # Build validation dataset (needed for COCO API)
    print("\nLoading COCO validation dataset...")
    dataset_val = build_dataset(
        image_set="val",
        args=eval_args,
        resolution=eval_args.resolution,
    )
    print(f"Validation dataset size: {len(dataset_val)} images")

    # Get COCO API
    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.load_predictions:
        # Load cached predictions
        predictions = load_predictions(output_dir, args.model)
    else:
        # Load model and run inference
        print(f"\nLoading {args.model} model...")
        start_time = time.time()
        model_cls = MODEL_CLASSES[args.model]
        rfdetr_model = model_cls()
        model_load_time = time.time() - start_time
        print(f"Model loaded in {model_load_time:.2f}s")

        # Get the underlying PyTorch model
        model = rfdetr_model.model.model
        model.eval()
        device = torch.device(args.device)
        model.to(device)

        # Build postprocessor
        print("Building postprocessor...")
        _, postprocess = build_criterion_and_postprocessors(eval_args)

        # Create data loader
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = DataLoader(
            dataset_val,
            eval_args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=eval_args.num_workers,
        )

        # Run inference and save predictions
        print("\nStarting inference...")
        predictions = run_inference(
            model,
            postprocess,
            data_loader_val,
            device,
            eval_args,
            output_dir,
            args.model,
        )

    # Evaluate predictions
    stats, coco_evaluator = evaluate_predictions(predictions, base_ds)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if "bbox" in stats:
        bbox = stats["bbox"]
        print(f"\nBounding Box Detection Metrics:")
        print(f"  AP (IoU=0.50:0.95) : {bbox['AP']:.4f}")
        print(f"  AP (IoU=0.50)      : {bbox['AP50']:.4f}")
        print(f"  AP (IoU=0.75)      : {bbox['AP75']:.4f}")
        print(f"  AP (small)         : {bbox['AP_small']:.4f}")
        print(f"  AP (medium)        : {bbox['AP_medium']:.4f}")
        print(f"  AP (large)         : {bbox['AP_large']:.4f}")
        print(f"\nAverage Recall:")
        print(f"  AR (max=1)         : {bbox['AR_1']:.4f}")
        print(f"  AR (max=10)        : {bbox['AR_10']:.4f}")
        print(f"  AR (max=100)       : {bbox['AR_100']:.4f}")
        print(f"  AR (small)         : {bbox['AR_small']:.4f}")
        print(f"  AR (medium)        : {bbox['AR_medium']:.4f}")
        print(f"  AR (large)         : {bbox['AR_large']:.4f}")

    # Save results if requested
    if args.save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"eval_results_{args.model}.json"
        results = {
            "model": args.model,
            "coco_path": str(args.coco_path),
            "device": args.device,
            "fp16": args.fp16,
            "stats": stats,
        }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
