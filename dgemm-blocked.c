const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 124
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define A(i, j) A[(i) + (j) * lda] // map A(i, j) to array A 
#define B(i, j) B[(i) + (j) * lda] // map B(i, j) to array B
#define C(i, j) C[(i) + (j) * lda] // map C(i, j) to array C

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//     // For each row i of A
//     for (int i = 0; i < M; ++i) {
//         // For each column j of B
//         for (int j = 0; j < N; ++j) {
//             // Compute C(i,j)
//             double cij = C[i + j * lda];
//             for (int k = 0; k < K; ++k) {
//                 cij += A[i + k * lda] * B[k + j * lda];
//             }
//             C[i + j * lda] = cij;
//         }
//     }
// }
    // For each row i of A
    for (int j = 0; j < M; ++j) {
        // For each column k of B
        for (int k = 0; k < K; ++k) {
            // // Compute C(i,j)
            double bkj = B(k, j);
            for (int i = 0; i < N; i++) {
                C(i, j) += A(i, k) * bkj;
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. 
 
 This implementation improves performance by reducing the number of cache misses 
 that occur when accessing memory, as blocks of data are stored in cache and reused
 multiple times before being evicted.
  */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/*
When the do_block function is called, it needs to know the starting 
position of the current block within the matrices A, B, and C in order 
to correctly perform the dgemm operation. The expressions A + i + k * lda, 
B + k + j * lda, and C + i + j * lda are used to calculate the starting 
position of the current block within each of the matrices.
*/