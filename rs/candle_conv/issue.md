Candle's convolution operations on CPU are quite slow, compared to Pytorch.

# Some numbers

Run configuration:
- batch_size = 2
- in_channels = 3
- width = 320
- height = 320
- out_channels = 16
- kernel_size = 3
- stride = 1
- padding = 1
- 1000 iterations, with 100 iters warmup

CPU benchmarks are noisy, so run variance is large.

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

# What do?

`im2col` seems to be a naive baseline.
