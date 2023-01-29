const char* dgemm_desc = "Naive, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < n; ++i) {
        // For each column j of B
        for (int j = 0; j < n; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * n];
            for (int k = 0; k < n; k++) {
                cij += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n = cij;
        }
    }
}



// void square_dgemm(int n, double* A, double* B, double* C) {
//     // For each row i of A
//     int cpj = 0;
//     for (int j = 0; j < n; ++j) {
//         // For each column j of B
//         for (int p = 0; p < n; ++p) {
//             // Compute C(i,j)
//             double bpj = B[p + j * n];
//             for (int i = 0; i < n; i++) {
//                 cpj += A[i + p * n] * bpj;
//             }
//             C[p + j * n] = cpj;
//         }
//     }
// }
