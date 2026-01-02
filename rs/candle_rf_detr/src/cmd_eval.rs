//! Eval subcommand for RF-DETR COCO evaluation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use clap::Args;

use crate::cmd_predict::{load_model, predict_image_raw, RawPrediction};
use crate::coco_eval::{self, CocoEvaluator, ImagePrediction};
use crate::config::RfDetrConfig;
use crate::model::RfDetr;
use crate::Which;

/// Arguments for the eval subcommand
#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Path to COCO dataset root (should contain 'annotations' and 'val2017' folders).
    #[arg(long)]
    pub coco_path: PathBuf,

    /// Output directory for evaluation results.
    #[arg(long, short, default_value = "eval_output")]
    pub output_dir: PathBuf,

    /// Load cached predictions instead of running inference.
    #[arg(long, short)]
    pub load_predictions: bool,
}

/// Convert raw prediction to ImagePrediction format for COCO evaluation.
fn raw_to_image_prediction(image_id: i64, raw: RawPrediction) -> ImagePrediction {
    ImagePrediction {
        image_id,
        scores: raw.scores,
        labels: raw.labels,
        boxes: raw.boxes,
    }
}

/// Run inference on all COCO validation images.
fn run_inference(
    model: &RfDetr,
    config: &RfDetrConfig,
    device: &Device,
    coco_dataset: &coco_eval::CocoDataset,
    coco_path: &PathBuf,
    pred_dir: &PathBuf,
) -> anyhow::Result<HashMap<i64, ImagePrediction>> {
    let val_images_dir = coco_path.join("val2017");
    if !val_images_dir.exists() {
        anyhow::bail!(
            "COCO val2017 images directory not found: {:?}",
            val_images_dir
        );
    }

    // Create predictions directory
    std::fs::create_dir_all(pred_dir)?;

    let num_images = coco_dataset.images.len();
    println!("\nRunning inference on {} images...", num_images);

    let mut predictions = HashMap::new();
    let start_time = Instant::now();
    let mut processed = 0;

    for (idx, img_info) in coco_dataset.images.iter().enumerate() {
        let image_path = val_images_dir.join(&img_info.file_name);

        if !image_path.exists() {
            eprintln!("Warning: Image not found: {:?}, skipping", image_path);
            continue;
        }

        // Run inference
        let image_path_str = image_path.to_string_lossy().to_string();
        let (raw_pred, _h, _w) = predict_image_raw(model, &image_path_str, config, device)?;

        // Convert to ImagePrediction
        let pred = raw_to_image_prediction(img_info.id, raw_pred);

        // Save prediction to disk immediately
        coco_eval::save_prediction(&pred, pred_dir)?;

        predictions.insert(img_info.id, pred);
        processed += 1;

        // Progress update every 100 images
        if (idx + 1) % 100 == 0 || idx == num_images - 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let imgs_per_sec = processed as f32 / elapsed;
            println!(
                "  [{}/{}] Processed {} images ({:.1} img/s)",
                idx + 1,
                num_images,
                processed,
                imgs_per_sec
            );
        }
    }

    let total_time = start_time.elapsed().as_secs_f32();
    println!(
        "\nInference completed in {:.1}s ({:.1} img/s)",
        total_time,
        processed as f32 / total_time
    );
    println!("Saved {} predictions to: {:?}", predictions.len(), pred_dir);

    Ok(predictions)
}

/// Run the eval subcommand
pub fn run(
    args: &EvalArgs,
    which: Which,
    model_path: PathBuf,
    device: &Device,
) -> anyhow::Result<()> {
    println!("============================================================");
    println!("RF-DETR COCO Evaluation (Rust)");
    println!("============================================================");
    println!("Model variant: {:?}", which);
    println!("Model path: {:?}", model_path);
    println!("Device: {:?}", device);
    println!("COCO path: {:?}", args.coco_path);
    println!("Output dir: {:?}", args.output_dir);
    println!("Load predictions: {}", args.load_predictions);
    println!("============================================================");

    // Verify COCO path
    let ann_file = args
        .coco_path
        .join("annotations")
        .join("instances_val2017.json");
    if !ann_file.exists() {
        anyhow::bail!(
            "COCO annotations not found: {:?}\n\
            Expected directory structure:\n  \
            {{coco_path}}/\n    \
            annotations/\n      \
            instances_val2017.json\n    \
            val2017/\n      \
            *.jpg",
            ann_file
        );
    }

    // Load COCO ground truth
    println!("\nLoading COCO ground truth annotations...");
    let coco_dataset = coco_eval::load_coco_annotations(&ann_file)?;

    // Get model name for predictions directory
    // Use "rust_" prefix to distinguish from Python predictions
    let model_name = format!("{:?}", which).to_lowercase();
    let pred_dir = args
        .output_dir
        .join(format!("predictions_rust_{}", model_name));

    // Load or generate predictions
    let predictions = if args.load_predictions {
        // Load cached predictions
        if !pred_dir.exists() {
            anyhow::bail!(
                "Predictions directory not found: {:?}\n\
                Run evaluation without --load-predictions first to generate predictions.",
                pred_dir
            );
        }
        println!("\nLoading predictions from: {:?}", pred_dir);
        coco_eval::load_predictions(&pred_dir)?
    } else {
        // Load model
        let config = which.config();
        println!("\nModel config:");
        println!("  Resolution: {}", config.resolution);
        println!("  Hidden dim: {}", config.hidden_dim);
        println!("  Decoder layers: {}", config.dec_layers);
        println!("  Num classes: {}", config.num_classes);

        println!("\nLoading model from: {:?}", model_path);
        let start = Instant::now();
        let model = load_model(&model_path, &config, device)?;
        println!("Model loaded in {:?}", start.elapsed());

        // Run inference
        run_inference(
            &model,
            &config,
            device,
            &coco_dataset,
            &args.coco_path,
            &pred_dir,
        )?
    };

    // Run COCO evaluation
    let evaluator = CocoEvaluator::new(&coco_dataset);
    let metrics = evaluator.evaluate(&predictions);

    // Print results
    println!("\n============================================================");
    println!("EVALUATION RESULTS");
    println!("============================================================");
    CocoEvaluator::print_summary(&metrics);

    // Save results
    std::fs::create_dir_all(&args.output_dir)?;
    let results_file = args
        .output_dir
        .join(format!("eval_results_rust_{}.json", model_name));
    let results = serde_json::json!({
        "model": model_name,
        "coco_path": args.coco_path.to_string_lossy(),
        "implementation": "rust",
        "stats": {
            "bbox": {
                "AP": metrics.ap,
                "AP50": metrics.ap50,
                "AP75": metrics.ap75,
                "AP_small": metrics.ap_small,
                "AP_medium": metrics.ap_medium,
                "AP_large": metrics.ap_large,
                "AR_1": metrics.ar_1,
                "AR_10": metrics.ar_10,
                "AR_100": metrics.ar_100,
                "AR_small": metrics.ar_small,
                "AR_medium": metrics.ar_medium,
                "AR_large": metrics.ar_large,
            }
        }
    });
    let file = std::fs::File::create(&results_file)?;
    serde_json::to_writer_pretty(file, &results)?;
    println!("\nEvaluation metrics saved to: {:?}", results_file);

    println!("\n============================================================");
    println!("Evaluation complete!");
    println!("============================================================");

    Ok(())
}
