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


// // Without caching 
// void square_dgemm(int n, double* A, double* B, double* C) {
//     // For each row i of A
//     for (int j = 0; j < n; ++j) {
//         // For each column k of B
//         for (int k = 0; k < n; ++k) {
//             // // Compute C(i,j)
//             for (int i = 0; i < n; i++) {
//                 C(i, j) += A(i, k) * B(k, j);
//             }
//         }
//     }
// }


// With caching
void square_dgemm(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int j = 0; j < n; ++j) {
        // For each column k of B
        for (int k = 0; k < n; ++k) {
            // // Compute C(i,j)
            double bkj = B(k, j);
            for (int i = 0; i < n; i++) {
                C(i, j) += A(i, k) * bkj;
            }
        }
    }
}