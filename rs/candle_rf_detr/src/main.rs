//! RF-DETR Object Detection with Candle
//!
//! This binary provides inference and evaluation capabilities for RF-DETR models.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod coco_classes;
mod coco_eval;
mod config;
mod detection;
mod dino2;
mod eval;
mod model;
mod pos_enc;
mod predict;
mod preprocess;
mod projector;
mod query_embed;
mod transformer;

use candle_core::{Device, Result};
use clap::{Parser, Subcommand, ValueEnum};

use crate::config::RfDetrConfig;

/// Select the compute device
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// RF-DETR model variants
#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum Which {
    Nano,
    Small,
    Medium,
    Base,
    Large,
}

impl Which {
    pub fn config(&self) -> RfDetrConfig {
        match self {
            Which::Nano => RfDetrConfig::nano(),
            Which::Small => RfDetrConfig::small(),
            Which::Medium => RfDetrConfig::medium(),
            Which::Base => RfDetrConfig::base(),
            Which::Large => RfDetrConfig::large(),
        }
    }

    fn default_weights(&self) -> &'static str {
        match self {
            Which::Nano => "rf-detr-nano.safetensors",
            Which::Small => "rf-detr-small.safetensors",
            Which::Medium => "rf-detr-medium.safetensors",
            Which::Base => "rf-detr-base.safetensors",
            Which::Large => "rf-detr-large.safetensors",
        }
    }
}

/// Subcommands for the CLI
#[derive(Subcommand, Debug)]
enum Command {
    /// Run object detection on a single image
    Predict(predict::PredictArgs),

    /// Evaluate on COCO validation dataset
    Eval(eval::EvalArgs),
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, global = true)]
    cpu: bool,

    /// Path to model weights, in safetensors format.
    #[arg(long, global = true)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::Small, global = true)]
    which: Which,

    #[command(subcommand)]
    command: Command,
}

impl Args {
    fn model_path(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                // Try to use HuggingFace Hub to download the model
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("roboflow/rf-detr".to_string());
                let filename = self.which.default_weights();
                match api.get(filename) {
                    Ok(path) => path,
                    Err(_) => {
                        // Fall back to local file
                        std::path::PathBuf::from(filename)
                    }
                }
            }
        };
        Ok(path)
    }
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Get device
    let device = device(args.cpu)?;
    println!("Using device: {:?}", device);

    // Get model path
    let model_path = args.model_path()?;

    match &args.command {
        Command::Predict(predict_args) => {
            predict::run(predict_args, args.which, model_path, &device)
        }
        Command::Eval(eval_args) => eval::run(eval_args, args.which, model_path, &device),
    }
}
