import torch
import torch.nn as nn
import time


def test_conv2d_cuda():
    """Test conv2d operation on CUDA with timing and assertions."""
    print("Running PyTorch conv2d test on CUDA...")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create deterministic input: batch_size=2, channels=3, height=32, width=32
    batch_size, in_channels, height, width = 2, 3, 32, 32

    # Create deterministic input tensor with hardcoded values for reproducibility
    input_tensor = torch.zeros(batch_size, in_channels, height, width, device=device)

    # Fill with deterministic patterns for each channel
    for b in range(batch_size):
        for c in range(in_channels):
            for h in range(height):
                for w in range(width):
                    # Create a deterministic pattern based on batch, channel, and position
                    value = (b + 1) * 0.1 + (c + 1) * 0.01 + (h * width + w) * 0.001
                    input_tensor[b, c, h, w] = value

    print(
        f"Input tensor statistics - Mean: {input_tensor.mean().item():.4f}, Std: {input_tensor.std().item():.4f}"
    )
    print(
        f"Input range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]"
    )

    # Create conv2d layer: 3 input channels, 16 output channels, 3x3 kernel, stride=1, padding=1
    out_channels, kernel_size, stride, padding = 16, 3, 1, 1
    conv_layer = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding
    ).to(device)

    # Warm up GPU
    for _ in range(10):
        with torch.no_grad():
            _ = conv_layer(input_tensor)

    torch.cuda.synchronize()

    # Time the operation
    num_runs = 100
    start_time = time.time()

    # No backprop
    with torch.no_grad():
        for _ in range(num_runs):
            output = conv_layer(input_tensor)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average conv2d time over {num_runs} runs: {avg_time:.3f} ms")

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


def main():
    test_conv2d_cuda()


if __name__ == "__main__":
    main()
