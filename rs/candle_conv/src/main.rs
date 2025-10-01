use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2dConfig, Module, VarBuilder, VarMap};
use std::time::Instant;

/// Create deterministic test tensor with given dimensions and device.
///
/// # Arguments
/// * `dims` - Tuple of dimensions (batch_size, channels, height, width)
/// * `device` - Device to place tensor on
///
/// # Returns
/// * `Result<Tensor>` - Deterministic tensor filled with structured pattern
fn test_tensor(dims: (usize, usize, usize, usize), device: &Device) -> Result<Tensor> {
    let (batch_size, channels, height, width) = dims;

    // Create deterministic input tensor with hardcoded values for reproducibility
    let mut input_data = vec![0.0f32; batch_size * channels * height * width];

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Create a deterministic pattern based on batch, channel, and position
                    let value = (b + 1) as f32 * 0.1
                        + (c + 1) as f32 * 0.01
                        + (h * width + w) as f32 * 0.001;
                    let idx = b * channels * height * width + c * height * width + h * width + w;
                    input_data[idx] = value;
                }
            }
        }
    }

    let input_tensor = Tensor::from_vec(input_data, (batch_size, channels, height, width), device)?;

    Ok(input_tensor)
}

// TODO make this into an actual test? but running via main is easier.
fn test_conv2d_cuda() -> Result<()> {
    println!("Running Candle conv2d test on CUDA...");

    // Create CUDA device
    let device = Device::new_cuda(0)?;
    // let device = Device::Cpu;
    println!("Using device: {device:?}");

    // Create deterministic input: batch_size=2, channels=3, height=32, width=32
    let (batch_size, in_channels, height, width) = (2, 3, 320, 320);
    let dims = (batch_size, in_channels, height, width);

    let input_tensor = test_tensor(dims, &device)?;

    // Calculate and print input statistics
    let input_mean: f32 = input_tensor.mean_all()?.to_scalar()?;
    let input_variance: f32 = input_tensor.var(0)?.mean_all()?.to_scalar()?;
    let input_std = input_variance.sqrt();
    let input_min: f32 = input_tensor.min_all()?.to_scalar()?;
    let input_max: f32 = input_tensor.max_all()?.to_scalar()?;

    println!(
        "Input tensor statistics - Mean: {:.4}, Std: {:.4}",
        input_mean, input_std
    );
    println!("Input range: [{:.4}, {:.4}]", input_min, input_max);

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

    // Warm up GPU
    for _ in 0..10 {
        let _ = conv_layer.forward(&input_tensor)?;
    }
    device.synchronize()?;

    // Time the operation
    let num_runs = 100;
    let start_time = Instant::now();

    let mut output = None;
    for _ in 0..num_runs {
        output = Some(conv_layer.forward(&input_tensor)?);
    }
    device.synchronize()?;

    let duration = start_time.elapsed();
    let avg_time = duration.as_millis() as f64 / num_runs as f64;

    let output = output.unwrap();

    println!("Input shape: {:?}", input_tensor.shape());
    println!("Output shape: {:?}", output.shape());
    println!(
        "Average conv2d time over {} runs: {:.3} ms",
        num_runs, avg_time
    );

    // Assertions to check outputs make sense
    let expected_output_shape = &[batch_size, out_channels, height, width];
    assert_eq!(
        output.shape().dims(),
        expected_output_shape,
        "Expected shape {:?}, got {:?}",
        expected_output_shape,
        output.shape().dims()
    );

    // Check that output is not all zeros (conv should produce meaningful results)
    let output_sum = output.sum_all()?.to_scalar::<f32>()?;
    assert!(
        output_sum.abs() > 1e-6,
        "Output should not be all zeros, got sum: {}",
        output_sum
    );

    // Check that output values are finite
    let output_data = output.flatten_all()?.to_vec1::<f32>()?;
    for (i, &val) in output_data.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output should contain only finite values, found {} at index {}",
            val,
            i
        );
    }

    // With deterministic input, we can verify the output is reproducible
    // Run the convolution again to ensure determinism
    let output2 = conv_layer.forward(&input_tensor)?;

    // Check if outputs are close (allowing for small floating point differences)
    let diff = output.sub(&output2)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-6,
        "Output should be deterministic - max difference: {}",
        max_diff
    );

    // Check that output has reasonable magnitude (adjusted for deterministic input)
    let output_mean = output.mean_all()?.to_scalar::<f32>()?;
    let output_variance = output.var(0)?.mean_all()?.to_scalar::<f32>()?;
    let output_std = output_variance.sqrt();
    let output_min = output.min_all()?.to_scalar::<f32>()?;
    let output_max = output.max_all()?.to_scalar::<f32>()?;

    // With our deterministic input pattern, we expect non-zero outputs
    assert!(
        output_std > 0.001,
        "Output standard deviation {:.6} is too small - convolution may not be working",
        output_std
    );

    // Verify the output makes sense given our input pattern
    // Our input increases monotonically, so we expect the conv output to reflect this structure
    assert!(
        output_mean.abs() > 0.001,
        "Output mean {:.6} is too close to zero for our structured input",
        output_mean
    );

    println!(
        "Output statistics - Mean: {:.6}, Std: {:.6}",
        output_mean, output_std
    );
    println!("Output range: [{:.6}, {:.6}]", output_min, output_max);
    println!(
        "✓ All assertions passed! Convolution is deterministic and produces expected results."
    );

    Ok(())
}

fn just_conv() -> Result<()> {
    let device = Device::new_cuda(0)?;
    // let device = Device::Cpu;

    // Create deterministic input: batch_size=2, channels=3, height=32, width=32
    let (batch_size, in_channels, height, width) = (2, 3, 320, 320);
    let dims = (batch_size, in_channels, height, width);

    let input_tensor = test_tensor(dims, &device)?;

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
    // warmup
    for _ in 0..10 {
        let _ = conv_layer.forward(&input_tensor)?;
    }
    device.synchronize()?;

    let start = Instant::now();
    let _ = conv_layer.forward(&input_tensor)?;

    device.synchronize()?;
    let elapsed = start.elapsed();
    println!("op finished in {elapsed:?}");

    Ok(())
}

fn main() -> Result<()> {
    // test_conv2d_cuda()
    just_conv()
}
