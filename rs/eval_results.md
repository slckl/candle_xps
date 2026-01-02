   Running `target/release/candle_rf_detr --which nano eval --coco-path ../datasets/coco`
Using device: Cuda(CudaDevice(DeviceId(1)))
rfdetr-nano.safetensors [00:00:14] [████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] 116.28 MiB/116.28 MiB 5.36 MiB/s (0s)============================================================
RF-DETR COCO Evaluation (Rust)
============================================================
Model variant: Nano
Device: Cuda(CudaDevice(DeviceId(1)))
COCO path: "../datasets/coco"
Output dir: "eval_output"
Load predictions: false
============================================================

Loading COCO ground truth annotations...
Loaded COCO annotations: 5000 images, 36781 annotations, 80 categories

Model config:
Resolution: 384
Hidden dim: 256
Decoder layers: 2
Num classes: 91

Model loaded in 32.073544ms

Running inference on 5000 images...
[100/5000] Processed 100 images (17.8 img/s)
[200/5000] Processed 200 images (16.4 img/s)
[300/5000] Processed 300 images (17.0 img/s)
[400/5000] Processed 400 images (17.3 img/s)
[500/5000] Processed 500 images (16.8 img/s)
[600/5000] Processed 600 images (17.0 img/s)
[700/5000] Processed 700 images (17.1 img/s)
[800/5000] Processed 800 images (17.2 img/s)
[900/5000] Processed 900 images (17.3 img/s)
[1000/5000] Processed 1000 images (17.3 img/s)
[1100/5000] Processed 1100 images (17.4 img/s)
[1200/5000] Processed 1200 images (17.4 img/s)
[1300/5000] Processed 1300 images (17.4 img/s)
[1400/5000] Processed 1400 images (17.4 img/s)
[1500/5000] Processed 1500 images (17.4 img/s)
[1600/5000] Processed 1600 images (17.4 img/s)
[1700/5000] Processed 1700 images (17.3 img/s)
[1800/5000] Processed 1800 images (17.2 img/s)
[1900/5000] Processed 1900 images (17.2 img/s)
[2000/5000] Processed 2000 images (17.1 img/s)
[2100/5000] Processed 2100 images (17.0 img/s)
[2200/5000] Processed 2200 images (17.0 img/s)
[2300/5000] Processed 2300 images (16.9 img/s)
[2400/5000] Processed 2400 images (16.8 img/s)
[2500/5000] Processed 2500 images (16.7 img/s)
[2600/5000] Processed 2600 images (16.7 img/s)
[2700/5000] Processed 2700 images (16.6 img/s)
[2800/5000] Processed 2800 images (16.4 img/s)
[2900/5000] Processed 2900 images (16.3 img/s)
[3000/5000] Processed 3000 images (16.2 img/s)
[3100/5000] Processed 3100 images (16.0 img/s)
[3200/5000] Processed 3200 images (15.9 img/s)
[3300/5000] Processed 3300 images (15.8 img/s)
[3400/5000] Processed 3400 images (15.8 img/s)
[3500/5000] Processed 3500 images (15.8 img/s)
[3600/5000] Processed 3600 images (15.7 img/s)
[3700/5000] Processed 3700 images (15.7 img/s)
[3800/5000] Processed 3800 images (15.7 img/s)
[3900/5000] Processed 3900 images (15.6 img/s)
[4000/5000] Processed 4000 images (15.6 img/s)
[4100/5000] Processed 4100 images (15.6 img/s)
[4200/5000] Processed 4200 images (15.5 img/s)
[4300/5000] Processed 4300 images (15.5 img/s)
[4400/5000] Processed 4400 images (15.4 img/s)
[4500/5000] Processed 4500 images (15.4 img/s)
[4600/5000] Processed 4600 images (15.4 img/s)
[4700/5000] Processed 4700 images (15.3 img/s)
[4800/5000] Processed 4800 images (15.3 img/s)
[4900/5000] Processed 4900 images (15.3 img/s)
[5000/5000] Processed 5000 images (15.2 img/s)

Inference completed in 328.1s (15.2 img/s)
Saved 5000 predictions to: "eval_output/predictions_rust_nano"

Running COCO evaluation...
5000 images in ground truth
5000 images with predictions
80 categories
IoU thresholds: ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"]

============================================================
EVALUATION RESULTS
============================================================

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.653
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.500
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.239
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.359
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.572
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.860

Evaluation metrics saved to: "eval_output/eval_results_rust_nano.json"

============================================================
Evaluation complete!
============================================================
