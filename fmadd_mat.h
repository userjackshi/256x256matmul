#include <immintrin.h>

namespace cache_op {
static inline __attribute__((always_inline)) void fmadd_mat(const double* __restrict A,
                                                                    const double* __restrict B, 
                                                                    double* __restrict C) {
    /* process C += AxB in chunks of 4x8 to maximize register usage
    * each ymm register holds 4 doubles so we will need two for each row
    * we will use registers ymm0-ymm7 for the accumulators of C
    */
    for (int a_row = 0; a_row < 128; a_row += 4) {
        for (int b_col = 0; b_col < 128; b_col += 8) {

            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
            ymm0 = _mm256_setzero_pd(); // first four doubles of first row
            ymm1 = _mm256_setzero_pd(); // second four doubles of first row
            ymm2 = _mm256_setzero_pd(); // ... 
            ymm3 = _mm256_setzero_pd(); 
            ymm4 = _mm256_setzero_pd();   
            ymm5 = _mm256_setzero_pd();   
            ymm6 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd(); // second four doubles of fourth row

            // load pointers to the i-th row of A and j-th column of B
            const double* A_ptr = &A[a_row*128];
            const double* B_ptr = &B[b_col*128]; // b_col/8 is the current strip and 128*8 is the size of the strip

            /* NB: custom formatting for B: columns are stored contiguously in panel layout with width 8 */

            for (int accum = 0; accum < 128; accum++) { // sum up 128 grids calculated by the outer product
                
                // load 4 values of A. each register holds one singular value i.e. ymmx = [ai, ai, ai, ai]
                __m256d ymm8, ymm9, ymm10, ymm11;
                ymm8 = _mm256_set1_pd(*A_ptr); // value of row i+0
                ymm9 = _mm256_set1_pd(*(A_ptr+128)); // value of row i+1 (since there are 128 numbers inbetween)
                ymm10 = _mm256_set1_pd(*(A_ptr+256)); // ...
                ymm11 = _mm256_set1_pd(*(A_ptr+384)); // value of row i+3
                A_ptr++; // move onto the next batch (column)

                // load current row of B; since 4x8, requires 2 registers
                __m256d ymm12, ymm13; 
                ymm12 = _mm256_load_pd(B_ptr);
                ymm13 = _mm256_load_pd(B_ptr+4);
                // efficiently access next row after only 8 elements due to panel layout (no need to traverse through 128 elems)
                B_ptr += 8; // next panel (same columns, next row)

                // now update the accumulators

                // calculate outer product of column from A and row from B and add it to the running total in C
                ymm0 = _mm256_fmadd_pd(ymm8, ymm12, ymm0); // C[0,0-3] <- A[0] * B[0-3]
                ymm1 = _mm256_fmadd_pd(ymm8, ymm13, ymm1); // C[0,4-7] <- A[0] * B[4-7]

                ymm2 = _mm256_fmadd_pd(ymm9, ymm12, ymm2); // etc...
                ymm3 = _mm256_fmadd_pd(ymm9, ymm13, ymm3);

                ymm4 = _mm256_fmadd_pd(ymm10, ymm12, ymm4);
                ymm5 = _mm256_fmadd_pd(ymm10, ymm13, ymm5);

                ymm6 = _mm256_fmadd_pd(ymm11, ymm12, ymm6);
                ymm7 = _mm256_fmadd_pd(ymm11, ymm13, ymm7); // C[3, 4-7] <- A[3] * B[4-7]
            }

            // store values back into C from the registers. Normal matrix layout.
            double* C_ptr = &C[128*(a_row+0)+b_col];
            _mm256_storeu_pd(C_ptr, _mm256_add_pd(_mm256_loadu_pd(C_ptr), ymm0)); // accumulates the value at C_ptr with ymm0
            _mm256_storeu_pd(C_ptr+4, _mm256_add_pd(_mm256_loadu_pd(C_ptr+4), ymm1)); // accumulates the value at C_ptr+4 with ymm0
            
            C_ptr = &C[128*(a_row+1)+b_col]; // onto the next row
            _mm256_storeu_pd(C_ptr, _mm256_add_pd(_mm256_loadu_pd(C_ptr), ymm2)); 
            _mm256_storeu_pd(C_ptr+4, _mm256_add_pd(_mm256_loadu_pd(C_ptr+4), ymm3));
            
            C_ptr = &C[128*(a_row+2)+b_col];
            _mm256_storeu_pd(C_ptr, _mm256_add_pd(_mm256_loadu_pd(C_ptr), ymm4)); 
            _mm256_storeu_pd(C_ptr+4, _mm256_add_pd(_mm256_loadu_pd(C_ptr+4), ymm5));
            
            C_ptr = &C[128*(a_row+3)+b_col];
            _mm256_storeu_pd(C_ptr, _mm256_add_pd(_mm256_loadu_pd(C_ptr), ymm6)); 
            _mm256_storeu_pd(C_ptr+4, _mm256_add_pd(_mm256_loadu_pd(C_ptr+4), ymm7));
        }
    }
}

static inline __attribute__((always_inline)) void neg_mat(double* __restrict A) {
    // load matrix into contiguous buffer [{i0j0, i0j1, i0j2, i0j3}, {i0j4...i0j7}, ..., {i127j123...i127j127}]
    __m256d identity = _mm256_setzero_pd(); // matrix of zeroes
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j+=4) {
            __m256d load_buffer_A = _mm256_loadu_pd(&A[128*i+j]);
            _mm256_store_pd(&A[128*i+j], _mm256_sub_pd(identity, load_buffer_A)); // 0-A = -A
        }
    }
}

/* NB: input matrices are 256x256. Each new row is accessed by 256*i. Output matrix is 128x128.
 * Use: to cover the entire 256x256 matrix, call functions on pointers starting at each four 128x128 block.
 */

// NB: source matrix should be 256x256, not 128x128. Output matrix C is 128x128.
// Only process a 128x128 chunk.

/* NB: for standard matrix layout
* i.e., {{1,2},{3,4}} -> [1,2,3,4]
* prototype: do operation on matrix A, B and store in C
*/


static inline __attribute__((always_inline)) void add_mat_a(const double* __restrict A, 
                                                                    const double* __restrict B, 
                                                                    double* __restrict C) {
    // load matrix into contiguous buffer [{i0j0, i0j1, i0j2, i0j3}, {i0j4...i0j7}, ..., {i127j123...i127j127}]
    for (int i = 0; i < 128; i++) { // per row
        for (int j = 0; j < 128; j+=4) { // per 4 values to load
            __m256d load_buffer_A = _mm256_loadu_pd(&A[256*i+j]);
            __m256d load_buffer_B = _mm256_loadu_pd(&B[256*i+j]);
            __m256d r = _mm256_add_pd(load_buffer_A, load_buffer_B); // A + B
            _mm256_store_pd(&C[128*i+j], r); // stores value into C contiguously
        }
    }
}

static inline __attribute__((always_inline)) void sub_mat_a(const double* __restrict A, 
                                                                    const double* __restrict B, 
                                                                    double* __restrict C) {
    // load matrix into contiguous buffer [{i0j0, i0j1, i0j2, i0j3}, {i0j4...i0j7}, ..., {i127j123...i127j127}]
    for (int i = 0; i < 128; i++) { // per row
        for (int j = 0; j < 128; j+=4) { // per 4 values to load
            __m256d load_buffer_A = _mm256_loadu_pd(&A[256*i+j]);
            __m256d load_buffer_B = _mm256_loadu_pd(&B[256*i+j]);
            __m256d r = _mm256_sub_pd(load_buffer_A, load_buffer_B); // A - B
            _mm256_store_pd(&C[128*i+j], r); // stores value into C contiguously
        }
    }
}

static inline __attribute__((always_inline)) void store_mat_a(const double* __restrict A,  
                                                                    double* __restrict B) {
    // load matrix into contiguous buffer [{i0j0, i0j1, i0j2, i0j3}, {i0j4...i0j7}, ..., {i127j123...i127j127}]
    for (int i = 0; i < 128; i++) { // per row
        for (int j = 0; j < 128; j+=4) { // per 4 values to load
            __m256d load_buffer_A = _mm256_loadu_pd(&A[256*i+j]);
            _mm256_store_pd(&B[128*i+j], load_buffer_A); // stores A into B
        }
    }
}

/* NB: for panel matrices (matrix B in fmadd_mat) 
* i.e., with width 2, {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}} -> strip 1 = [(1,2),5,6,9,10,13,14]; circled = width
* strip 2 = [3,4,7,8,11,12,15,16]; store = [strip 1 | strip 2] contiguously 
* prototype: do operation on matrix A, B and store in C. C is in panel format.
*/

static inline __attribute__((always_inline)) void add_mat_b(const double* __restrict A, 
                                                                    const double* __restrict B, 
                                                                    double* __restrict C) {
    for (int j = 0; j < 128; j+=8) { // per column, 8 values to load
        for (int i = 0; i < 128; i++) { // per row
            __m256d load_buffer_A_0 = _mm256_loadu_pd(&A[256*i+j]);
            __m256d load_buffer_B_0 = _mm256_loadu_pd(&B[256*i+j]);
            __m256d load_buffer_A_1 = _mm256_loadu_pd(&A[256*i+j+4]);
            __m256d load_buffer_B_1 = _mm256_loadu_pd(&B[256*i+j+4]);
            __m256d r0 = _mm256_add_pd(load_buffer_A_0, load_buffer_B_0); // A + B first 4 values
            __m256d r1 = _mm256_add_pd(load_buffer_A_1, load_buffer_B_1); // A + B last 4 values
            _mm256_store_pd(C, r0); // stores first 4 values into C contiguously
            _mm256_store_pd(C+4, r1); // stores next 4 values into C contiguously
            C+=8; // next block
        }
    }
}

static inline __attribute__((always_inline)) void sub_mat_b(const double* __restrict A, 
                                                                    const double* __restrict B, 
                                                                    double* __restrict C) {
    for (int j = 0; j < 128; j+=8) { // per column, 8 values to load
        for (int i = 0; i < 128; i++) { // per row
            __m256d load_buffer_A_0 = _mm256_loadu_pd(&A[256*i+j]);
            __m256d load_buffer_B_0 = _mm256_loadu_pd(&B[256*i+j]);
            __m256d load_buffer_A_1 = _mm256_loadu_pd(&A[256*i+j+4]);
            __m256d load_buffer_B_1 = _mm256_loadu_pd(&B[256*i+j+4]);
            __m256d r0 = _mm256_sub_pd(load_buffer_A_0, load_buffer_B_0); // A - B first 4 values
            __m256d r1 = _mm256_sub_pd(load_buffer_A_1, load_buffer_B_1); // A - B last 4 values
            _mm256_store_pd(C, r0); // stores first 4 values into C contiguously
            _mm256_store_pd(C+4, r1); // stores next 4 values into C contiguously
            C+=8; // next block
        }
    }
}

static inline __attribute__((always_inline)) void store_mat_b(const double* __restrict A,  
                                                                    double* __restrict B) {
    for (int j = 0; j < 128; j+=8) { // per column, 8 values to load
        for (int i = 0; i < 128; i++) { // per row
            __m256d load_buffer_A_0 = _mm256_loadu_pd(&A[256*i+j]);
            __m256d load_buffer_A_1 = _mm256_loadu_pd(&A[256*i+j+4]);
            _mm256_store_pd(B, load_buffer_A_0); // stores first 4 values into C contiguously
            _mm256_store_pd(B+4, load_buffer_A_1); // stores next 4 values into C contiguously
            B+=8; // next block
        }
    }
}
};
