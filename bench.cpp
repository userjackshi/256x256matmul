// Naive triple loop vs. Strassen + AVX2 microkernel, 256x256 doubles.
// Reports median wall time, effective GFLOP/s (2n^3 / t), and speedup.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "strassen.h"

static double A[256][256], B[256][256], Cnaive[256][256];

// Fair baseline: same compiler, same flags; ikj order so -O3 can autovectorize.
static void naive_matmul(const double a[256][256], const double b[256][256],
                         double c[256][256]) {
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++) c[i][j] = 0.0;
    for (int i = 0; i < 256; i++)
        for (int k = 0; k < 256; k++) {
            const double aik = a[i][k];
            for (int j = 0; j < 256; j++) c[i][j] += aik * b[k][j];
        }
}

// Keep results observable so the optimizer cannot delete the work.
static volatile double g_sink;

template <typename F>
static double median_ms(F&& fn, int warmup, int trials) {
    for (int i = 0; i < warmup; i++) fn();
    std::vector<double> t(trials);
    for (int i = 0; i < trials; i++) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        t[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(t.begin(), t.end());
    return t[trials / 2];
}

int main() {
    srand(42);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++) {
            A[i][j] = (rand() / (double)RAND_MAX - 0.5) * 4.0;
            B[i][j] = (rand() / (double)RAND_MAX - 0.5) * 4.0;
        }

    const double flops = 2.0 * 256.0 * 256.0 * 256.0;  // classical count
    const int warmup = 10, trials = 101;

    double t_naive = median_ms(
        [&] {
            naive_matmul(A, B, Cnaive);
            g_sink = Cnaive[255][255];
        },
        warmup, trials);

    double t_kernel = median_ms(
        [&] {
            double(*C)[256] = strassen::multiply_matrices(A, B);
            g_sink = C[255][255];
        },
        warmup, trials);

    double t_kernel_sum = median_ms(
        [&] { g_sink = strassen::multiply_matrices_sum(A, B); }, warmup, trials);

    printf("naive  (ikj, -O3)      : %8.3f ms   %6.2f GFLOP/s\n", t_naive,
           flops / (t_naive * 1e6));
    printf("strassen+AVX2 (matrix) : %8.3f ms   %6.2f GFLOP/s   speedup %.2fx\n",
           t_kernel, flops / (t_kernel * 1e6), t_naive / t_kernel);
    printf("strassen+AVX2 (frob^2) : %8.3f ms   %6.2f GFLOP/s   speedup %.2fx\n",
           t_kernel_sum, flops / (t_kernel_sum * 1e6), t_naive / t_kernel_sum);
    return 0;
}
