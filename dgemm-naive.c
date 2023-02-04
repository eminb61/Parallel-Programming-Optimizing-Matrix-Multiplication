const char* dgemm_desc = "Naive, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * The best ordering is j, k, i.
 */
void square_dgemm(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int j = 0; j < n; j++) {
        // For each column j of B
        for (int k = 0; k < n; k++) {
            // Compute C(i,j)
            double bkj = B[k + j * n];
            for (int i = 0; i < n; i++) {
                C[i + j * n] += A[i + k * n] * bkj;
            }
            // C[i + j * n] = cij;
        }
    }
}



// void square_dgemm(int n, double* A, double* B, double* C) {
//     // For each row i of A
//     for (int k = 0; k < n; k++) {
//         // For each column j of B
//         for (int j = 0; j < n; j++) {
//             // Compute C(i,j)
//             double bkj = B[k + j * n];
//             for (int i = 0; i < n; i++) {
//                 C[i + j * n] += A[i + k * n] * bkj;
//             }
//             // C[i + j * n] = cij;
//         }
//     }
// }
