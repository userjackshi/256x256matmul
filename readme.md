# 256x256 matmul kernel

Efficiently computes the product of two 256x256 matrices of doubles. Uses one layer of Strassen and handles the resulting seven 128x128 products with an outer-product microkernel: a 4x8 register block held in eight AVX2 FMA accumulators, with B packed into a width-8 panel layout so every load streams contiguously.

## API

Use `strassen::`. Inputs take row-major `double[256][256]`.

- `multiply_matrices_sum(a, b)` returns the squared Frobenius norm of the product without materializing it.
- `multiply_matrices(a, b)` returns a pointer to the product in thread_local static storage which the next call on the same thread overwrites.
- `multiply_matrices_into(a, b, c_out)` copies the product into caller-owned storage.

Scratch buffers are thread_local, so concurrent calls from different threads are safe.

## Benchmark

Requires an x86-64 CPU with AVX2 and FMA.
Use flags `-O3 -march=native -funroll-loops`. Benchmark reports median wall time, effective GFLOP/s against the classical 2n^3 count, and speedup.
