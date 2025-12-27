# RF-DETR Object Detection with Candle

This crate provides an implementation of [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow DETR) object detection using the [Candle](https://github.com/huggingface/candle) ML framework in Rust.

## Overview

RF-DETR is a state-of-the-art real-time object detection model that combines:
- **DINOv2 backbone** with windowed attention for efficient feature extraction
- **DETR-style transformer decoder** with deformable attention
- **Two-stage detection** for improved accuracy

## Model Variants

| Variant | Resolution | Hidden Dim | Decoder Layers | Parameters |
|---------|------------|------------|----------------|------------|
| Nano    | 384        | 256        | 2              | ~30M       |
| Small   | 512        | 256        | 3              | ~80M       |
| Medium  | 576        | 256        | 4              | ~90M       |
| Base    | 560        | 256        | 3              | ~90M       |
| Large   | 560        | 384        | 3              | ~300M      |

## Usage

```bash
# Run inference on an image
cargo run --release -- --model rf-detr-medium.safetensors image.jpg

# Specify model variant
cargo run --release -- --which small --model rf-detr-small.safetensors image.jpg

# Adjust detection threshold
cargo run --release -- --confidence-threshold 0.3 --nms-threshold 0.5 image.jpg

# Run on CPU
cargo run --release -- --cpu --model rf-detr-medium.safetensors image.jpg
```

## Command Line Options

```
Usage: candle_rf_detr [OPTIONS] [IMAGES]...

Arguments:
  [IMAGES]...  Input images to process

Options:
      --cpu                                    Run on CPU rather than on GPU
      --tracing                                Enable tracing (generates a trace-timestamp.json file)
      --model <MODEL>                          Model weights, in safetensors format
      --which <WHICH>                          Which model variant to use [default: medium]
      --confidence-threshold <THRESHOLD>       Threshold for confidence level [default: 0.5]
      --nms-threshold <NMS_THRESHOLD>          Threshold for NMS [default: 0.5]
      --num-select <NUM_SELECT>                Number of top detections [default: 300]
      --legend-size <LEGEND_SIZE>              Size for the legend, 0 means no legend [default: 14]
  -h, --help                                   Print help
  -V, --version                                Print version
```

## Exporting Model Weights

RF-DETR models need to be exported to safetensors format from PyTorch. Use the provided Python script:

```bash
cd ../../py/rfdetr

# Export a model
uv run export_safetensors.py --model medium --output rf-detr-medium.safetensors

# List model keys (for debugging)
uv run export_safetensors.py --list-keys --model medium
```

## Architecture Notes

### Current Implementation Status

This implementation provides the core RF-DETR architecture components:

- ✅ DINOv2 backbone (encoder blocks, patch embeddings, position embeddings)
- ✅ Multi-scale feature projector
- ✅ Transformer decoder layers
- ✅ Detection heads (classification + bounding box)
- ✅ Two-stage proposal generation
- ✅ Post-processing (NMS, box format conversion)
- ⚠️ Cross-attention uses standard attention (not deformable)

### Deformable Attention

The original RF-DETR uses **Multi-Scale Deformable Attention** (MSDeformAttn) which requires custom CUDA kernels. This implementation uses standard cross-attention as a fallback, which may affect accuracy compared to the original PyTorch model.

For optimal performance matching the original model, consider:
1. Using ONNX runtime with the exported ONNX model
2. Implementing deformable attention as a custom Candle operation

## Key Differences from YOLOv8

Unlike YOLOv8 which is a single-shot detector:

| Aspect | YOLOv8 | RF-DETR |
|--------|--------|---------|
| Architecture | CNN-based | Transformer-based |
| Detection | Anchor-free grid | Query-based |
| NMS | Required post-process | Built into model |
| Feature extraction | FPN/PAN | DINOv2 + Projector |
| Attention | None | Self + Cross attention |
