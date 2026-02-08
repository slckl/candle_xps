// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

mod cmd_eval;
mod cmd_predict;
mod coco_classes;
mod config;
#[cfg(test)]
mod debug;
mod detection;
mod model;
mod preprocess;

use candle_core::{Device, Result};
use clap::{Parser, Subcommand, ValueEnum};

use crate::config::RfDetrConfig;

/// Picks best available compute device, if not cpu.
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
    LargeDeprecated,
    SegPreview,
    SegNano,
    SegSmall,
    SegMedium,
    SegLarge,
    SegXLarge,
    Seg2XLarge,
}

impl Which {
    pub fn config(&self) -> RfDetrConfig {
        match self {
            Which::Nano => RfDetrConfig::nano(),
            Which::Small => RfDetrConfig::small(),
            Which::Medium => RfDetrConfig::medium(),
            Which::Base => RfDetrConfig::base(),
            Which::Large => RfDetrConfig::large(),
            Which::LargeDeprecated => RfDetrConfig::large_deprecated(),
            Which::SegPreview => RfDetrConfig::seg_preview(),
            Which::SegNano => RfDetrConfig::seg_nano(),
            Which::SegSmall => RfDetrConfig::seg_small(),
            Which::SegMedium => RfDetrConfig::seg_medium(),
            Which::SegLarge => RfDetrConfig::seg_large(),
            Which::SegXLarge => RfDetrConfig::seg_xlarge(),
            Which::Seg2XLarge => RfDetrConfig::seg_2xlarge(),
        }
    }

    fn default_weights(&self) -> &'static str {
        match self {
            Which::Nano => "rfdetr-nano.safetensors",
            Which::Small => "rfdetr-small.safetensors",
            Which::Medium => "rfdetr-medium.safetensors",
            Which::Base => "rfdetr-base.safetensors",
            Which::Large => "rfdetr-large.safetensors",
            Which::LargeDeprecated => "rfdetr-large-deprecated.safetensors",
            Which::SegPreview => "rfdetr-seg-preview.safetensors",
            Which::SegNano => "rfdetr-seg-nano.safetensors",
            Which::SegSmall => "rfdetr-seg-small.safetensors",
            Which::SegMedium => "rfdetr-seg-medium.safetensors",
            Which::SegLarge => "rfdetr-seg-large.safetensors",
            Which::SegXLarge => "rfdetr-seg-xlarge.safetensors",
            Which::Seg2XLarge => "rfdetr-seg-2xlarge.safetensors",
        }
    }
}

/// Subcommands for the CLI
#[derive(Subcommand, Debug)]
enum Command {
    /// Run object detection on a single image
    Predict(cmd_predict::PredictArgs),
    /// Evaluate on COCO validation dataset
    Eval(cmd_eval::EvalArgs),
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
    #[arg(long, value_enum, default_value_t = Which::Nano, global = true)]
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
                let api = api.model("slckl/candle-rf-detr".to_string());
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
            cmd_predict::run(predict_args, args.which, model_path, &device)
        }
        Command::Eval(eval_args) => cmd_eval::run(eval_args, args.which, model_path, &device),
    }
}
