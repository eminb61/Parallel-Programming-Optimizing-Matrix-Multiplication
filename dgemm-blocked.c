const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE 64
#define BLOCK_SIZE_L1 64
#define BLOCK_SIZE_L2 256

#define min(a, b) (((a) < (b)) ? (a) : (b))



static inline void kernel4by4(int lda, int K, double* A, double* B, double* C) {
    double c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
    c00 = C[0];
    c10 = C[1];
    c20 = C[2];
    c30 = C[3];
    c01 = C[lda];
    c11 = C[1 + lda];
    c21 = C[2 + lda];
    c31 = C[3 + lda];
    c02 = C[2 * lda];
    c12 = C[1 + 2 * lda];
    c22 = C[2 + 2 * lda];
    c32 = C[3 + 2 * lda];
    c03 = C[3 * lda];
    c13 = C[1 + 3 * lda];
    c23 = C[2 + 3 * lda];
    c33 = C[3 + 3 * lda];
    for (int k = 0; k < K; k++) {
        double a0x, a1x, a2x, a3x, bx0, bx1, bx2, bx3;
        a0x = A[k * lda];
        a1x = A[1 + k * lda];
        a2x = A[2 + k * lda];
        a3x = A[3 + k * lda];
        bx0 = B[k];
        bx1 = B[k + lda];
        bx2 = B[k + 2 * lda];
        bx3 = B[k + 3 * lda];
        
        c00 += a0x * bx0;
        c01 += a0x * bx1;
        c02 += a0x * bx2;
        c03 += a0x * bx3;
        c10 += a1x * bx0;
        c11 += a1x * bx1;
        c12 += a1x * bx2;
        c13 += a1x * bx3;
        c20 += a2x * bx0;
        c21 += a2x * bx1;
        c22 += a2x * bx2;
        c23 += a2x * bx3;
        c30 += a3x * bx0;
        c31 += a3x * bx1;
        c32 += a3x * bx2;
        c33 += a3x * bx3;
    }
    C[0] = c00;
    C[1] = c10;
    C[2] = c20;
    C[3] = c30;
    C[lda] = c01;
    C[1 + lda] = c11;
    C[2 + lda] = c21;
    C[3 + lda] = c31;
    C[2 * lda] = c02;
    C[1 + 2 * lda] = c12;
    C[2 + 2 * lda] = c22;
    C[3 + 2 * lda] = c32;
    C[3 * lda] = c03;
    C[1 + 3 * lda] = c13;
    C[2 + 3 * lda] = c23;
    C[3 + 3 * lda] = c33;
}

static inline void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    int Nedge = N % 4;
    int Medge = M % 4;
    int Nmax = N - Nedge;
    int Mmax = M - Medge;
    
    // For each column j of B
    for (int j = 0; j < Nmax; j+=4) {
        for (int i = 0; i < Mmax; i+=4) {
            // Compute C(i,j)
            kernel4by4(lda, K, A + i, B + j * lda, C + i + j*lda);
        }
    }
    if ( Medge != 0 ) {
        for (int i = Mmax; i < M; i++) {
            for (int u = 0; u < N; u++) {
                double ciu = C[i + u * lda];
                for (int k = 0; k < K; k++) {
                    ciu += A[i + k * lda] * B[k + u * lda]; 
                }
                C[i + u * lda] = ciu;
            }
        }
    }

    if ( Nedge != 0 ) {
       for (int j = Nmax; j < N; j++) {
            for (int i = 0; i < Mmax; i++ ) {
                double cij = C[i + j * lda];
                for (int k = 0; k < K; k++) {
                    cij += A[i + k * lda] * B[k + j * lda]; 
                }
                C[i + j * lda] = cij;
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int v = 0; v < lda; v += BLOCK_SIZE_L2) {
        int lda_j = v + min(BLOCK_SIZE_L2, lda - v);
        for (int w = 0; w < lda; w += BLOCK_SIZE_L2) {
            int lda_k = w + min(BLOCK_SIZE_L2, lda - w);
            for (int u = 0; u < lda; u += BLOCK_SIZE_L2) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int lda_i = u + min(BLOCK_SIZE_L2, lda - u);
                for (int j = v; j < lda_j; j += BLOCK_SIZE_L1) {
                    int N = min(BLOCK_SIZE_L1, lda_j - j);
                    // For each block-column of B
                    for (int k = w; k < lda_k; k += BLOCK_SIZE_L1) {
                        int K = min(BLOCK_SIZE_L1, lda_k - k);
                        // Accumulate block dgemms into block of C
                        for (int i = u; i < lda_i; i += BLOCK_SIZE_L1) {
                            // Correct block dimensions if block "goes off edge of" the matrix
                            int M = min(BLOCK_SIZE_L1, lda_i - i);
                            // Perform individual block dgemm
                            do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                        }
                    }
                }
            }
        }
    }
}

// void square_dgemm(int lda, double* A, double* B, double* C) {
//     // For each block-row of A
//     for (int i = 0; i < lda; i += BLOCK_SIZE) {
//         // For each block-column of B
//         for (int j = 0; j < lda; j += BLOCK_SIZE) {
//             // Accumulate block dgemms into block of C
//             for (int k = 0; k < lda; k += BLOCK_SIZE) {
//                 // Correct block dimensions if block "goes off edge of" the matrix
//                 int M = min(BLOCK_SIZE, lda - i);
//                 int N = min(BLOCK_SIZE, lda - j);
//                 int K = min(BLOCK_SIZE, lda - k);
//                 // Perform individual block dgemm
//                 do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
//             }
//         }
//     }
// }