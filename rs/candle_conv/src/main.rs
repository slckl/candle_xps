use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2dConfig, Module, VarBuilder, VarMap};
use std::time::{Duration, Instant};

fn just_conv() -> Result<()> {
    // let device = Device::new_cuda(0)?;
    let device = Device::Cpu;

    // Create deterministic input: batch_size=2, channels=3, height=32, width=32
    let (batch_size, in_channels, height, width) = (2, 3, 320, 320);

    // let input_tensor = test_tensor(dims, &device)?;
    let input_tensor = Tensor::from_vec(
        vec![1.0f32; batch_size * in_channels * height * width],
        (batch_size, in_channels, height, width),
        &device,
    )?;

    // Create conv2d layer: 3 input channels, 16 output channels, 3x3 kernel, stride=1, padding=1
    let (out_channels, kernel_size, stride, padding) = (16, 3, 1, 1);

    // Create VarMap and VarBuilder for the conv layer
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let conv_config = Conv2dConfig {
        padding,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    let conv_layer = candle_nn::conv2d_no_bias(
        in_channels,
        out_channels,
        kernel_size,
        conv_config,
        vb.pp("conv"),
    )?;
    const ITERS: usize = 1000;
    const WARMUP: usize = 100;
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    let mut total = Duration::ZERO;
    for i in 0..ITERS + WARMUP {
        // println!("---");
        let start = Instant::now();
        let _ = conv_layer.forward(&input_tensor)?;
        device.synchronize()?;
        let elapsed = start.elapsed();
        if i > WARMUP {
            total += elapsed;
            if elapsed < min {
                min = elapsed;
            }
            if elapsed > max {
                max = elapsed;
            }
        }
        // println!("conv2d {:?}", start.elapsed());
    }
    device.synchronize()?;

    println!(
        "{ITERS} iters, min/avg/max: [{:?} {:?} {:?}]",
        min,
        total / ITERS as u32,
        max
    );

    Ok(())
}

fn main() -> Result<()> {
    // test_conv2d_cuda()
    just_conv()
}
