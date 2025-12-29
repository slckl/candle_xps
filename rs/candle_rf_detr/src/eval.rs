//! Eval subcommand for RF-DETR COCO evaluation.

use std::path::PathBuf;

use candle_core::Device;
use clap::Args;

use crate::coco_eval::{self, CocoEvaluator};
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

/// Run the eval subcommand
pub fn run(
    args: &EvalArgs,
    which: Which,
    model_path: PathBuf,
    device: &Device,
) -> anyhow::Result<()> {
    println!("============================================================");
    println!("RF-DETR COCO Evaluation");
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
    let model_name = format!("{:?}", which).to_lowercase();
    let pred_dir = args.output_dir.join(format!("predictions_{}", model_name));

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
        // TODO: Run inference to generate predictions
        anyhow::bail!(
            "Inference not yet implemented. Use --load-predictions to load cached predictions \
            from the Python evaluation (py/rfdetr/eval_output/)."
        )
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
        .join(format!("eval_results_{}.json", model_name));
    let results = serde_json::json!({
        "model": model_name,
        "coco_path": args.coco_path.to_string_lossy(),
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
