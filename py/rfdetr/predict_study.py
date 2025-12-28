import argparse
import os
import time
from typing import Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.transforms import functional as F

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


def dump_tensor(tensor: torch.Tensor, path: str, name: str):
    """Dump a tensor to a text file in a human-readable format."""
    filepath = os.path.join(path, f"{name}.txt")
    t = tensor.detach().cpu().float()

    with open(filepath, "w") as f:
        f.write(f"# Shape: {list(t.shape)}\n")
        f.write(f"# Dtype: {tensor.dtype}\n")
        f.write(
            f"# Min: {t.min().item():.6f}, Max: {t.max().item():.6f}, Mean: {t.mean().item():.6f}\n"
        )
        f.write(f"# Sum: {t.sum().item():.6f}\n\n")

        # Flatten and write values
        flat = t.flatten().numpy()
        # Write first 1000 values for quick inspection, then full data
        f.write(f"# First 1000 values (of {len(flat)} total):\n")
        for i, val in enumerate(flat[:1000]):
            f.write(f"{val:.6f}\n")

        if len(flat) > 1000:
            f.write(f"\n# ... ({len(flat) - 1000} more values) ...\n")

    # Also save as numpy binary for exact reproduction
    np.save(os.path.join(path, f"{name}.npy"), t.numpy())
    print(f"  Dumped {name}: shape={list(t.shape)}")


def predict_with_debug(model, image, threshold=0.5, debug_dir: Optional[str] = None):
    """
    Perform prediction with optional intermediate output dumping.

    Args:
        model: The RFDETR model instance
        image: PIL Image to process
        threshold: Detection confidence threshold
        debug_dir: If provided, dump intermediate outputs to this directory

    Returns:
        sv.Detections object with detection results
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug outputs will be saved to: {debug_dir}")

    # Get the underlying PyTorch model
    pytorch_model = model.model.model
    pytorch_model.eval()

    # Step 1: Preprocess image
    img_tensor = F.to_tensor(image)
    h_orig, w_orig = img_tensor.shape[1:]

    if debug_dir:
        dump_tensor(img_tensor, debug_dir, "01_input_image_raw")

    img_tensor = img_tensor.to(model.model.device)
    img_tensor = F.normalize(img_tensor, model.means, model.stds)

    if debug_dir:
        dump_tensor(img_tensor, debug_dir, "02_input_image_normalized")

    img_tensor = F.resize(img_tensor, (model.model.resolution, model.model.resolution))

    if debug_dir:
        dump_tensor(img_tensor, debug_dir, "03_input_image_resized")

    batch_tensor = img_tensor.unsqueeze(0)

    # Step 2: Run through backbone encoder
    with torch.inference_mode():
        backbone = pytorch_model.backbone[0]

        # Encoder forward pass
        encoder_output = backbone.encoder(batch_tensor)

        if debug_dir:
            if isinstance(encoder_output, (list, tuple)):
                for i, feat in enumerate(encoder_output):
                    dump_tensor(feat, debug_dir, f"04_backbone_encoder_output_{i}")
            else:
                dump_tensor(encoder_output, debug_dir, "04_backbone_encoder_output")

        # Step 3: Projector forward pass
        projector_output = backbone.projector(encoder_output)

        if debug_dir:
            if isinstance(projector_output, (list, tuple)):
                for i, feat in enumerate(projector_output):
                    dump_tensor(feat, debug_dir, f"05_backbone_projector_output_{i}")
            else:
                dump_tensor(projector_output, debug_dir, "05_backbone_projector_output")

        # Step 4: Position encoding
        # Get position embeddings from backbone joiner
        joiner = pytorch_model.backbone
        srcs = projector_output

        # Create position embeddings
        position_encoding = joiner[1]
        poss = []
        for feat in srcs:
            # Create a simple mask (all False = all valid)
            b, c, h, w = feat.shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)

            # In export mode, forward_export expects just the mask tensor
            # (model.export() switches forward to forward_export)
            pos = position_encoding(mask, align_dim_orders=False).to(feat.dtype)
            poss.append(pos)

        if debug_dir:
            for i, pos in enumerate(poss):
                dump_tensor(pos, debug_dir, f"06_position_encoding_{i}")

        # Step 5: Transformer forward pass
        # Prepare inputs for transformer
        transformer = pytorch_model.transformer

        # Use only one group for inference
        num_queries = pytorch_model.num_queries
        refpoint_embed_weight = pytorch_model.refpoint_embed.weight[:num_queries]
        query_feat_weight = pytorch_model.query_feat.weight[:num_queries]

        if debug_dir:
            dump_tensor(refpoint_embed_weight, debug_dir, "07_refpoint_embed")
            dump_tensor(query_feat_weight, debug_dir, "08_query_feat")

        # Run transformer
        # Create masks (None for export-style inference)
        masks = None

        hs, ref_unsigmoid, hs_enc, ref_enc = transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )

        if debug_dir:
            if hs is not None:
                dump_tensor(hs, debug_dir, "09_transformer_decoder_hidden_states")
                dump_tensor(
                    ref_unsigmoid, debug_dir, "10_transformer_decoder_references"
                )
            if hs_enc is not None:
                dump_tensor(hs_enc, debug_dir, "11_transformer_encoder_hidden_states")
            if ref_enc is not None:
                dump_tensor(ref_enc, debug_dir, "12_transformer_encoder_references")

        # Step 6: Classification and box prediction heads
        # Note: In export mode, hs has shape [batch_size, num_queries, hidden_dim]
        # (not [num_layers, batch_size, num_queries, hidden_dim] like in training mode)
        if hs is not None:
            # Bbox prediction with reparameterization
            if pytorch_model.bbox_reparam:
                outputs_coord_delta = pytorch_model.bbox_embed(hs)

                if debug_dir:
                    dump_tensor(
                        outputs_coord_delta, debug_dir, "13_bbox_embed_raw_output"
                    )

                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = (
                    outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                )
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (pytorch_model.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            if debug_dir:
                dump_tensor(outputs_coord, debug_dir, "14_bbox_predictions")

            # Classification
            outputs_class = pytorch_model.class_embed(hs)

            if debug_dir:
                dump_tensor(outputs_class, debug_dir, "15_class_logits")

            # In export mode, outputs are already [batch_size, num_queries, ...]
            # No need to index with [-1] like in training mode
            pred_logits = outputs_class
            pred_boxes = outputs_coord
        else:
            # Two-stage only mode
            pred_logits = transformer.enc_out_class_embed[0](hs_enc)
            pred_boxes = ref_enc

        if debug_dir:
            dump_tensor(pred_logits, debug_dir, "16_final_class_logits")
            dump_tensor(pred_boxes, debug_dir, "17_final_bbox_predictions")

        # Step 7: Postprocess
        predictions = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

        target_sizes = torch.tensor([[h_orig, w_orig]], device=model.model.device)
        results = model.model.postprocess(predictions, target_sizes=target_sizes)

        if debug_dir:
            dump_tensor(results[0]["scores"], debug_dir, "18_postprocess_scores")
            dump_tensor(results[0]["labels"], debug_dir, "19_postprocess_labels")
            dump_tensor(results[0]["boxes"], debug_dir, "20_postprocess_boxes")

    # Filter by threshold and create detections
    result = results[0]
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]

    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    detections = sv.Detections(
        xyxy=boxes.float().cpu().numpy(),
        confidence=scores.float().cpu().numpy(),
        class_id=labels.cpu().numpy(),
    )

    return detections


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR prediction with debug output")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        choices=list(MODEL_CLASSES.keys()),
        help=f"Model size to use. Available: {', '.join(MODEL_CLASSES.keys())} (default: small)",
    )
    parser.add_argument(
        "--debug-dir",
        "-d",
        type=str,
        default=None,
        help="Directory to dump intermediate outputs for debugging (default: None, no debug output)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="sample.jpg",
        help="Input image path (default: sample.jpg)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_time = time.time()
    model_cls = MODEL_CLASSES[args.model]
    model = model_cls()
    model_load_time = time.time() - start_time
    print(f"Model loading time: {model_load_time:.4f} seconds")

    # Don't optimize for inference when debugging - we need access to intermediate layers
    if args.debug_dir is None:
        start_time = time.time()
        model.optimize_for_inference()
        optimize_time = time.time() - start_time
        print(f"Model optimization time: {optimize_time:.4f} seconds")
    else:
        print("Skipping optimization (debug mode requires access to model internals)")
        model.model.model.eval()
        model.model.model.export()

    image = Image.open(args.image)
    start_time = time.time()

    if args.debug_dir:
        detections = predict_with_debug(
            model, image, threshold=args.threshold, debug_dir=args.debug_dir
        )
    else:
        detections = model.predict(image, threshold=args.threshold)

    predict_time = time.time() - start_time
    print(f"Model prediction time: {predict_time:.4f} seconds")

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    print(f"{len(labels)} objects detected:")
    for label in labels:
        print(f"  {label}")

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    # save image to output.jpg
    output_path = "sample.py.jpg"
    annotated_image.save(output_path)
    print(f"Annotated image saved to: {output_path}")
