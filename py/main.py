import time

import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function


def test_tensor(dims: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    """Create deterministic test tensor with given dimensions and device.

    Args:
        dims: Tuple of dimensions (batch_size, channels, height, width)
        device: Torch device to place tensor on

    Returns:
        torch.Tensor: Deterministic tensor filled with structured pattern
    """
    batch_size, channels, height, width = dims

    # Create deterministic input tensor with hardcoded values for reproducibility
    input_tensor = torch.zeros(batch_size, channels, height, width, device=device)

    # Fill with deterministic patterns for each channel
    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    # Create a deterministic pattern based on batch, channel, and position
                    value = (b + 1) * 0.1 + (c + 1) * 0.01 + (h * width + w) * 0.001
                    input_tensor[b, c, h, w] = value

    return input_tensor


def test_conv2d_cuda():
    """Test conv2d operation on CUDA with timing and assertions."""
    print("Running PyTorch conv2d test on CUDA...")

    device = torch.device("cpu")
    # device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create deterministic input: batch_size=2, channels=3, height=32, width=32
    batch_size, in_channels, height, width = 2, 3, 320, 320
    dims = (batch_size, in_channels, height, width)

    # input_tensor = test_tensor(dims, device)
    input_tensor = torch.ones(batch_size, in_channels, height, width, device=device)
    print(
        f"Input tensor statistics - Mean: {input_tensor.mean().item():.4f}, Std: {input_tensor.std().item():.4f}"
    )
    print(
        f"Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]"
    )

    # Create conv2d layer: 3 input channels, 16 output channels, 3x3 kernel, stride=1, padding=1
    out_channels, kernel_size, stride, padding = 16, 3, 1, 1
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    ).to(device)

    # Warm up
    warmup = 100
    for _ in range(warmup):
        with torch.no_grad():
            _ = conv_layer(input_tensor)

    # torch.cuda.synchronize()

    # Time the operation
    num_runs = 1000
    start_time = time.time()
    min_time = float("inf")
    max_time = float("-inf")

    # No backprop
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True,
    #     with_modules=True,
    # ) as prof:
    #     with record_function("conv2d_forward"):
    with torch.no_grad():
        for _ in range(num_runs):
            run_start = time.time()
            output = conv_layer(input_tensor)
            run_elapsed = time.time() - run_start
            min_time = min(min_time, run_elapsed)
            max_time = max(max_time, run_elapsed)

    # torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average conv2d time over {num_runs} runs: {avg_time:.3f} ms")
    print(f"Min conv2d time: {min_time * 1000:.3f} ms")
    print(f"Max conv2d time: {max_time * 1000:.3f} ms")

    # profiling kernels
    # Print detailed results
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export for visualization
    # prof.export_chrome_trace("conv2d_trace.json")
    # print("\nTrace exported to conv2d_trace.json - open in chrome://tracing")

    # Print stack traces for CUDA kernels
    # print("\n" + "=" * 80)
    # print("Detailed kernel information:")
    # print("=" * 80)
    # for evt in prof.key_averages():
    # if evt.device_type == torch.profiler.DeviceType.CUDA:
    # print(f"\nKernel: {evt.key}")
    # print(f"  Time: {evt.cuda_time_total / 1000:.3f} ms")
    # print(f"  Shape: {evt.input_shapes}")

    # Assertions to check outputs make sense
    expected_output_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_output_shape, (
        f"Expected shape {expected_output_shape}, got {output.shape}"
    )

    # Check that output is not all zeros (conv should produce meaningful results)
    assert not torch.allclose(output, torch.zeros_like(output)), (
        "Output should not be all zeros"
    )

    # Check that output values are finite
    assert torch.all(torch.isfinite(output)), "Output should contain only finite values"

    # With deterministic input, we can verify the output is reproducible
    # Run the convolution again to ensure determinism
    with torch.no_grad():
        output2 = conv_layer(input_tensor)

    assert torch.allclose(output, output2), (
        "Output should be deterministic - got different results on repeated runs"
    )

    # Check that output has reasonable magnitude (adjusted for deterministic input)
    output_std = output.std().item()
    output_mean = output.mean().item()

    # With our deterministic input pattern, we expect non-zero outputs
    assert output_std > 0.001, (
        f"Output standard deviation {output_std:.6f} is too small - convolution may not be working"
    )

    # Verify the output makes sense given our input pattern
    # Our input increases monotonically, so we expect the conv output to reflect this structure
    assert abs(output_mean) > 0.001, (
        f"Output mean {output_mean:.6f} is too close to zero for our structured input"
    )

    print(
        f"Output statistics - Mean: {output.mean().item():.6f}, Std: {output_std:.6f}"
    )
    print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(
        "✓ All assertions passed! Convolution is deterministic and produces expected results."
    )


def just_conv():
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create deterministic input: batch_size=2, channels=3, height=32, width=32
    batch_size, in_channels, height, width = 2, 3, 32, 32
    dims = (batch_size, in_channels, height, width)

    input_tensor = test_tensor(dims, device)

    # Create conv2d layer: 3 input channels, 16 output channels, 3x3 kernel, stride=1, padding=1
    out_channels, kernel_size, stride, padding = 16, 3, 1, 1
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    ).to(device)

    with torch.no_grad():
        _ = conv_layer(input_tensor)

    torch.cuda.synchronize()
    print("All gucci")


def main():
    test_conv2d_cuda()
    # just_conv()


if __name__ == "__main__":
    main()
