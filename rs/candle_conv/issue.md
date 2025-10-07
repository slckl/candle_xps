Candle's convolution operations on CPU are quite slow, compared to Pytorch.

# Some numbers

Conv2d run configuration:
- batch_size = 2
- in_channels = 3
- width = 320
- height = 320
- out_channels = 16
- kernel_size = 3
- stride = 1
- padding = 1
- 1000 iterations, with 100 iters warmup

CPU benchmarks are noisy, so run variance is large, but the pecking order of who's fastest on the given platform should be consistent enough.

Experiment code for pytorch: xxx
Experiment code for candle: xxx

## i7-12700h

| Lib           | Op                | Min     | Max     | Avg     |
| ------------- | ----------------- | ------- | --------| ------- |
| Candle 0.9.1  | Conv2d im2col     | 14.9 ms | 56.5 ms | 23.9 ms |
| Candle 0.9.1  | Conv2d non-im2col | 24.4 ms | 47.6 ms | 27.1 ms |
| Pytorch 2.8.0 | Conv2d            | 0.8 ms  | 4.9 ms  | 2.1 ms  |

## Ryzen 5900x

| Lib           | Op                | Min     | Max     | Avg     |
| ------------- | ----------------- | ------- | --------| ------- |
| Candle 0.9.1  | Conv2d im2col     | 13.9 ms | 17.3 ms | 15.4 ms |
| Candle 0.9.1  | Conv2d non-im2col | 14.7 ms | 19.0 ms | 16.7 ms |
| Pytorch 2.8.0 | Conv2d            | 0.6 ms  | 3.7 ms  | 2.0 ms  |

# Anatomy of a cpu conv call

There are 2 pathways for conv2d in candle cpu right now:
1. non-im2col or direct - a direct convolution implementation using for loops in pure rust.
2. im2col - where input/image tensor is converted to a columnar format which allows the convolution oepration to be expressed as a gemm call.

Both are slow right now.

The im2col version exploits a fast gemm kernel, but the overhead of im2col and post-kernel transformations kill any gains.

Sample Timings of an im2col conv2d kernel:
- im2col: 9.667183ms
- kernel setup: 824.975µs
- kernel exec: 1.134914ms
- copy_strided_src: 6.325587ms
- total conv2d 18.021546ms

# What do?

Looking at onednn documentation for inspiration:  https://github.com/uxlfoundation/oneDNN/blob/main/doc/primitives/convolution.md#algorithms

They seem to have :
- optimized direct convolution impl,
- fallback implicit gemm (which is also what cudnn does, afaik),
- specialized kernels.

`im2col` seems to be a naive baseline, does not look like any sota frameworks use it.
