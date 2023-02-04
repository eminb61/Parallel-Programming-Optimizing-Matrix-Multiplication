const char* dgemm_desc = "Naive, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * 
 * Index i is used to index the row of C and corresponding row of A. 
 * Index j is used to index the column of C and corresponding column of B. 
 * Index k is used to index the column of A and corresponding row of B.
 */

#define A(i, j) A[(i) + (j) * n] // map A(i, j) to array A 
#define B(i, j) B[(i) + (j) * n] // map B(i, j) to array B
#define C(i, j) C[(i) + (j) * n] // map C(i, j) to array C

void square_dgemm(int n, double* A, double* B, double* C) {
    // double cij;
    // For each row i of A
    for (int j = 0; j < n; ++j) {
        // For each column j of B
        for (int k = 0; k < n; ++k) {
            // // Compute B(k,j)
            // double bkj = B(k, j);
            for (int i = 0; i < n; i++) {
                // cij += A(i, k) * bkj;
                C(i, j) += A(i, k) * B(k, j);
            }
            // C(i, j) = cij;
        }
    }
}