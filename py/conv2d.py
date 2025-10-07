import time

import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function


def test_conv2d():
    device = torch.device("cpu")
    # device = torch.device("cuda")
    print(f"Running pytorch conv2d on device: {device}")

    # Create deterministic input: batch_size=2, channels=3, height=32, width=32
    batch_size, in_channels, height, width = 2, 3, 320, 320

    input_tensor = torch.ones(batch_size, in_channels, height, width, device=device)

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

    # Commented out profiling code, which lets us find out which kernels are used
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


def main():
    test_conv2d()


if __name__ == "__main__":
    main()
