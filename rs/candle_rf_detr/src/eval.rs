//! Eval subcommand for RF-DETR COCO evaluation.

use std::path::PathBuf;

use candle_core::Device;
use clap::Args;

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
}

/// Run the eval subcommand
pub fn run(
    args: &EvalArgs,
    which: Which,
    model_path: PathBuf,
    device: &Device,
) -> anyhow::Result<()> {
    println!("Eval subcommand (stub)");
    println!("  Model variant: {:?}", which);
    println!("  Model path: {:?}", model_path);
    println!("  Device: {:?}", device);
    println!("  COCO path: {:?}", args.coco_path);
    println!("  Output dir: {:?}", args.output_dir);

    anyhow::bail!("Eval subcommand not yet implemented")
}
