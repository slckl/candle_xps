import torch
import torch.nn as nn
import time


def test_conv2d_cuda():
    """Test conv2d operation on CUDA with timing and assertions."""
    print("Running PyTorch conv2d test on CUDA...")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create sample input: batch_size=2, channels=3, height=32, width=32
    batch_size, in_channels, height, width = 2, 3, 32, 32
    input_tensor = torch.randn(batch_size, in_channels, height, width, device=device)

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

    # Check that output has reasonable magnitude (not too large or too small)
    output_std = output.std().item()
    assert 0.01 < output_std < 100.0, (
        f"Output standard deviation {output_std:.3f} seems unreasonable"
    )

    print(
        f"Output statistics - Mean: {output.mean().item():.4f}, Std: {output_std:.4f}"
    )
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("✓ All assertions passed!")


def main():
    test_conv2d_cuda()


if __name__ == "__main__":
    main()
