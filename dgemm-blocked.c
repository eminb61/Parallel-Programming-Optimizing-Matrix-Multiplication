const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE 16
// #define BLOCK_SIZE_L1 64
// #define BLOCK_SIZE_L2 256

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define alpha( i,j ) A[ (j)*lda + (i) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (j)*lda + (i) ]   // map beta( i,j ) to array B
#define gamma( i,j ) C[ (j)*lda + (i) ]   // map gamma( i,j ) to array C

#define AA(i, j) A[i + lda * j]   // map alpha( i,j ) to array A
#define BB(i, j) B[i + lda * j]   // map beta( i,j ) to array B
#define CC(i, j) C[i + lda * j]   // map gamma( i,j ) to array C

#include<immintrin.h>
#include<string.h>
#include<stdio.h>

void pointerFuncA(double* iptr){
    /*Print the value pointed to by iptr*/
    printf("A \n");
    printf("First Element:  %f\n", iptr[0]);
    printf("991:  %f\n", iptr[990]);
    printf("992:  %f\n", iptr[991]);
    printf("Last Element:  %f\n", iptr[1023]);

    /*Print the address pointed to by iptr*/
    printf("Address of value: %p\n", (void*)iptr);

    /*Print the address of iptr itself*/
    // printf("Address of iptr: %p\n", (void*)&iptr);
}
void pointerFuncB(double* iptr){
    /*Print the value pointed to by iptr*/
    printf("B \n");
    printf("First Element:  %f\n", iptr[0]);
    printf("991:  %f\n", iptr[990]);
    printf("992:  %f\n", iptr[991]);
    printf("Last Element:  %f\n", iptr[1023]);

    /*Print the address pointed to by iptr*/
    printf("Address of value: %p\n", (void*)iptr);

    /*Print the address of iptr itself*/
    // printf("Address of iptr: %p\n", (void*)&iptr);
}

static void kernel4by4_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
{
  __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( 0,0 ) );
  __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( 0,1 ) );
  __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( 0,2 ) );
  __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( 0,3 ) );

  __m256d beta_p_j;
   	
  for ( int p=0; p<K; p++ ){
    /* load alpha( 0:3, p ) */
    __m256d alpha_0123_p = _mm256_loadu_pd( MP_A );

    /* load beta( p, 0 ); update gamma( 0:3, 0 ) */
    beta_p_j = _mm256_broadcast_sd( MP_B );
    gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );

    /* load beta( p, 1 ); update gamma( 0:3, 1 ) */
    beta_p_j = _mm256_broadcast_sd( MP_B+1 );
    gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );

    /* load beta( p, 2 ); update gamma( 0:3, 2 ) */
    beta_p_j = _mm256_broadcast_sd( MP_B+2 );
    gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );

    /* load beta( p, 3 ); update gamma( 0:3, 3 ) */
    beta_p_j = _mm256_broadcast_sd( MP_B+3 );
    gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );

    MP_A += 4;
    MP_B += 4;
  }

  /* Store the updated results.  This should be done more carefully since
     there may be an incomplete micro-tile. */
  _mm256_storeu_pd( &gamma(0,0), gamma_0123_0 );
  _mm256_storeu_pd( &gamma(0,1), gamma_0123_1 );
  _mm256_storeu_pd( &gamma(0,2), gamma_0123_2 );
  _mm256_storeu_pd( &gamma(0,3), gamma_0123_3 );
}

static inline void kernel4by4(int lda, int K, double *A, double *B,double *C) {
    /* Declare vector registers to hold 4x4 C and load them */
    __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( 0,0 ) );
    __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( 0,1 ) );
    __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( 0,2 ) );
    __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( 0,3 ) );

    for ( int p=0; p<K; p++ ){
    /* Declare vector register for load/broadcasting beta( p,j ) */
    __m256d beta_p_j;

    /* Declare a vector register to hold the current column of A and load
        it with the four elements of that column. */
    __m256d alpha_0123_p = _mm256_loadu_pd( &alpha( 0,p ) );

    /* Load/broadcast beta( p,0 ). */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 0) );

    /* update the first column of C with the current column of A times
        beta ( p,0 ) */
    gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );

    /* REPEAT for second, third, and fourth columns of C.  Notice that the 
        current column of A needs not be reloaded. */
    beta_p_j = _mm256_broadcast_sd( &beta( p, 1) );
    gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );

    beta_p_j = _mm256_broadcast_sd( &beta( p, 2) );
    gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );

    beta_p_j = _mm256_broadcast_sd( &beta( p, 3) );
    gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );
    }

    /* Store the updated results */
    _mm256_storeu_pd( &gamma(0,0), gamma_0123_0 );
    _mm256_storeu_pd( &gamma(0,1), gamma_0123_1 );
    _mm256_storeu_pd( &gamma(0,2), gamma_0123_2 );
    _mm256_storeu_pd( &gamma(0,3), gamma_0123_3 );
}

// static inline void kernel4by4(int lda, int K, double* A, double* B, double* C) {
//     double c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
//     c00 = C[0];
//     c10 = C[1];
//     c20 = C[2];
//     c30 = C[3];
//     c01 = C[lda];
//     c11 = C[1 + lda];
//     c21 = C[2 + lda];
//     c31 = C[3 + lda];
//     c02 = C[2 * lda];
//     c12 = C[1 + 2 * lda];
//     c22 = C[2 + 2 * lda];
//     c32 = C[3 + 2 * lda];
//     c03 = C[3 * lda];
//     c13 = C[1 + 3 * lda];
//     c23 = C[2 + 3 * lda];
//     c33 = C[3 + 3 * lda];
//     for (int k = 0; k < K; k++) {
//         double a0x, a1x, a2x, a3x, bx0, bx1, bx2, bx3;
//         a0x = A[k * lda];
//         a1x = A[1 + k * lda];
//         a2x = A[2 + k * lda];
//         a3x = A[3 + k * lda];
//         bx0 = B[k];
//         bx1 = B[k + lda];
//         bx2 = B[k + 2 * lda];
//         bx3 = B[k + 3 * lda];
        
//         c00 += a0x * bx0;
//         c01 += a0x * bx1;
//         c02 += a0x * bx2;
//         c03 += a0x * bx3;
//         c10 += a1x * bx0;
//         c11 += a1x * bx1;
//         c12 += a1x * bx2;
//         c13 += a1x * bx3;
//         c20 += a2x * bx0;
//         c21 += a2x * bx1;
//         c22 += a2x * bx2;
//         c23 += a2x * bx3;
//         c30 += a3x * bx0;
//         c31 += a3x * bx1;
//         c32 += a3x * bx2;
//         c33 += a3x * bx3;
//     }
//     C[0] = c00;
//     C[1] = c10;
//     C[2] = c20;
//     C[3] = c30;
//     C[lda] = c01;
//     C[1 + lda] = c11;
//     C[2 + lda] = c21;
//     C[3 + lda] = c31;
//     C[2 * lda] = c02;
//     C[1 + 2 * lda] = c12;
//     C[2 + 2 * lda] = c22;
//     C[3 + 2 * lda] = c32;
//     C[3 * lda] = c03;
//     C[1 + 3 * lda] = c13;
//     C[2 + 3 * lda] = c23;
//     C[3 + 3 * lda] = c33;
// }


// /* This routine performs a dgemm operation
//  *  C := C + A * B
//  * where A, B, and C are lda-by-lda matrices stored in column-major format.
//  * On exit, A and B maintain their input values. */
// void square_dgemm(int lda, double* A, double* B, double* C) {
//     // For each block-row of A
//     for (int v = 0; v < lda; v += BLOCK_SIZE_L2) {
//         int lda_j = v + min(BLOCK_SIZE_L2, lda - v);
//         for (int w = 0; w < lda; w += BLOCK_SIZE_L2) {
//             int lda_k = w + min(BLOCK_SIZE_L2, lda - w);
//             for (int u = 0; u < lda; u += BLOCK_SIZE_L2) {
//                 // Correct block dimensions if block "goes off edge of" the matrix
//                 int lda_i = u + min(BLOCK_SIZE_L2, lda - u);
//                 for (int j = v; j < lda_j; j += BLOCK_SIZE_L1) {
//                     int N = min(BLOCK_SIZE_L1, lda_j - j);
//                     // For each block-column of B
//                     for (int k = w; k < lda_k; k += BLOCK_SIZE_L1) {
//                         int K = min(BLOCK_SIZE_L1, lda_k - k);
//                         // Accumulate block dgemms into block of C
//                         for (int i = u; i < lda_i; i += BLOCK_SIZE_L1) {
//                             // Correct block dimensions if block "goes off edge of" the matrix
//                             int M = min(BLOCK_SIZE_L1, lda_i - i);
//                             // Perform individual block dgemm
//                             do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// static inline void copy_a (int lda, int K, double* a_src, double* a_dest) {
//   /* For each 4xK block-row of A */
//   for (int i = 0; i < K; ++i) 
//   {
//     *a_dest++ = *a_src;
//     *a_dest++ = *(a_src+1);
//     *a_dest++ = *(a_src+2);
//     *a_dest++ = *(a_src+3);
//     a_src += lda;
//   }
// }

// static inline void copy_b (int lda, int K, double* b_src, double* b_dest) {
//   double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
//   b_ptr0 = b_src;
//   b_ptr1 = b_ptr0 + lda;
//   b_ptr2 = b_ptr1 + lda;
//   b_ptr3 = b_ptr2 + lda;

//   for (int i = 0; i < K; ++i) 
//   {
//     *b_dest++ = *b_ptr0++;
//     *b_dest++ = *b_ptr1++;
//     *b_dest++ = *b_ptr2++;
//     *b_dest++ = *b_ptr3++;
//   }
// }

// void square_dgemm(int lda, double* A, double* B, double* C) {
//     // For each block-row of A
//     double *a_p, *b_p, *c_p;
//     for (int j = 0; j < lda; j += BLOCK_SIZE) {
//         // For each block-column of B
//         int N = min(BLOCK_SIZE, lda - j);
//         for (int k = 0; k < lda; k += BLOCK_SIZE) {
//             // Accumulate block dgemms into block of C
//             int K = min(BLOCK_SIZE, lda - k);
//             for (int i = 0; i < lda; i += BLOCK_SIZE) {
//                 // Correct block dimensions if block "goes off edge of" the matrix
//                 int M = min(BLOCK_SIZE, lda - i);
//                 // Perform individual block dgemm
//                 // do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
//                 a_p = &A[i + k * lda];
//                 b_p = &B[k + j * lda];
//                 c_p = &C[i + j * lda];
//                 double A_block[M*K], B_block[K*N];
//                 double *a_ptr, *b_ptr;

//                 int Nedge = N % 4;
//                 int Medge = M % 4;
//                 int Nmax = N - Nedge;
//                 int Mmax = M - Medge;
                
//                 // For each column j of B
//                 for (int j = 0; j < Nmax; j+=4) {
//                     b_ptr = &B_block[j*K];
//                     copy_b(lda, K, &b_p[j * lda], b_ptr);
//                     for (int i = 0; i < Mmax; i+=4) {
//                         // Compute C(i,j)
//                         a_ptr = &A_block[i*K];
//                         if (j == 0) copy_a(lda, K, &a_p[i], a_ptr);
//                         // kernel4by4(lda, K, &a_p[i], &b_p[j * lda], &c_p[i + j*lda]);
//                         kernel4by4_packed(lda, K, a_ptr, b_ptr, &c_p[i + j*lda]);
//                     }
//                 }
//                 if ( Medge != 0 ) {
//                     for (int i = Mmax; i < M; i++) {
//                         for (int u = 0; u < N; u++) {
//                             double ciu = c_p[i + u * lda];
//                             for (int k = 0; k < K; k++) {
//                                 ciu += a_p[i + k * lda] * b_p[k + u * lda]; 
//                             }
//                             c_p[i + u * lda] = ciu;
//                         }
//                     }
//                 }

//                 if ( Nedge != 0 ) {
//                 for (int j = Nmax; j < N; j++) {
//                         for (int i = 0; i < Mmax; i++ ) {
//                             double cij = c_p[i + j * lda];
//                             for (int k = 0; k < K; k++) {
//                                 cij += a_p[i + k * lda] * b_p[k + j * lda]; 
//                             }
//                             c_p[i + j * lda] = cij;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }


// static inline void copy_a (int lda, int K, double* a_src, double* a_dest) {
//   /* For each 4xK block-row of A */
//   for (int i = 0; i < K; ++i) 
//   {
//     *a_dest++ = *a_src;
//     *a_dest++ = *(a_src+1);
//     *a_dest++ = *(a_src+2);
//     *a_dest++ = *(a_src+3);
//     a_src += lda;
//   }
// }

// static inline void copy_b (int lda, int K, double* b_src, double* b_dest) {
//   double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
//   b_ptr0 = b_src;
//   b_ptr1 = b_ptr0 + lda;
//   b_ptr2 = b_ptr1 + lda;
//   b_ptr3 = b_ptr2 + lda;

//   for (int i = 0; i < K; ++i) 
//   {
//     *b_dest++ = *b_ptr0++;
//     *b_dest++ = *b_ptr1++;
//     *b_dest++ = *b_ptr2++;
//     *b_dest++ = *b_ptr3++;
//   }
// }

void PackMicroPanelA_MRxKC(int K, int lda, double *A, double *Atilde ){
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
  /* March through A in column-major order, packing into Atilde as we go. */
    /* Full row size micro-panel.*/
    for (int p=0; p<K; p++){
        for (int i=0; i<4; i++) {
            *Atilde++ = AA(i, p);
        }
    }
}

void PackMicroPanelA_MRxKC_Pad(int K, int Medge, int lda, double *A, double *Atilde ){
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
  /* March through A in column-major order, packing into Atilde as we go. */
    /* Full row size micro-panel.*/
    int Mpad = 4 - Medge;
    // printf("K: %d\n", K);
    // printf("Medge: %d\n", Medge);
    for (int p=0; p<K; p++){
        for (int i=0; i<Medge; i++) {
            // printf("%d, %d:  %f\n", i, p, AA(i, p));
            *Atilde++ = AA(i, p);
        }
        Atilde += Mpad;
    }
}

void PackBlockA_MCxKC(int M, int K, int lda, double *A, double *Atilde ){
/* Pack a  m x k block of A into a MC x KC buffer.   MC is assumed to
    be a multiple of MR.  The block is packed into Atilde a micro-panel
    at a time. If necessary, the last micro-panel is padded with rows
    of zeroes. */
    int Medge = M % 4;
    int Mmax = M - Medge;
    // printf("Medge: %d\n", Medge);
    // printf("Mmax: %d\n", Mmax);
    // printf("before pack \n");
    // printf("1:  %f\n", Atilde[0]);
    // printf("256:  %f\n", Atilde[255]);
    for (int i=0; i<Mmax; i+= 4){
        PackMicroPanelA_MRxKC(K, lda, &AA(i, 0), Atilde);
        Atilde += K * 4;
    }
    // printf("after pack \n");
    // printf("1:  %f\n", Atilde[0]);
    // printf("256:  %f\n", Atilde[255]);

    if (Medge != 0){
        // printf("In A pad \n");
        // for (int i=0; i<20; i++) {
        //     printf("%d:  %f\n", i, Atilde[i]);
        // }
        
        PackMicroPanelA_MRxKC_Pad(K, Medge, lda, &AA(Mmax, 0), Atilde);
        // printf("After A pad \n");
        // for (int i=0; i<16; i++) {
        //     printf("%d:  %f\n", i, Atilde[i]);
        // }
        Atilde += K * 4;
    }
}

void PackMicroPanelB_KCxNR_Pad(int K, int Nedge, int lda, double *B, double *Btilde){
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
    int Npad = 4 - Nedge;
    for (int p=0; p<K; p++){
        for (int j=0; j<Nedge; j++){
            *Btilde++ = BB(p, j);
        }
        Btilde += Npad;
    }
}

void PackMicroPanelB_KCxNR(int K, int lda, double *B, double *Btilde){
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
    for (int p=0; p<K; p++){
        for (int j=0; j<4; j++){
            *Btilde++ = BB(p, j);
        }
    }
}

void PackPanelB_KCxNC(int K, int N, int lda, double *B, double *Btilde){
/* Pack a k x n panel of B in to a KC x NC buffer.
   The block is copied into Btilde a micro-panel at a time. */
    int Nedge = N % 4;
    int Nmax = N - Nedge;
    for (int j=0; j<Nmax; j+= 4){
        PackMicroPanelB_KCxNR(K, lda, &BB(0, j), Btilde);
        Btilde += K * 4;
    }
    if (Nedge != 0){
        PackMicroPanelB_KCxNR_Pad(K, Nedge, lda, &BB(0, Nmax), Btilde);
        Btilde += K * 4;
    }
   
}

void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    // printf("lda: %d\n", lda);
    // printf("A first: %f\n", AA(0,0));
    // printf("B first: %f\n", BB(0,0));
    // printf("A last: %f\n", AA(30,30));
    // printf("B last: %f\n", BB(30,30));
    double *c_p;
    int blk = (lda + 4 - 1) / 4;
    int ldc = blk * 4;
    double *Ctilde = (double *) malloc(ldc * ldc * sizeof(double)); 
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
        // For each block-column of B
        int N = min(BLOCK_SIZE, lda - j);
        
        double *Btilde = (double *) calloc(BLOCK_SIZE * BLOCK_SIZE, sizeof(double)); // Target L3, KC * NC
        for (int k = 0; k < lda; k += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            int K = min(BLOCK_SIZE, lda - k);
            // pointerFuncB(Btilde);
            PackPanelB_KCxNC(K, N, lda, &BB(k, j), Btilde);
            // pointerFuncB(Btilde);
            double *Atilde = (double *) calloc(BLOCK_SIZE * BLOCK_SIZE, sizeof(double)); // Target L2, MC * KC
            
            for (int i = 0; i < lda; i += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                
                // pointerFuncA(Atilde);
                PackBlockA_MCxKC(M, K, lda, &AA(i, k), Atilde);
                // pointerFuncA(Atilde);
                // Perform individual block dgemm
                c_p = &Ctilde[i + j*ldc];
                for (int v = 0; v < N; v+=4) {
                    for (int u = 0; u < M; u+=4) {
                        // kernel4by4(lda, K, &a_p[i], &b_p[j * lda], &c_p[i + j*lda]);
                        kernel4by4_packed(ldc, K, &Atilde[u*K], &Btilde[v*K], &c_p[u + v*ldc]);
                    }
                }
            }
            free(Atilde);
        }
        free(Btilde);
    }
    if (ldc != lda) {
        for (int i = 0; i < lda; i ++){
            memcpy(C + lda * i, Ctilde + ldc * i, ldc * sizeof(double));
        }
    }else{
        memcpy(C, Ctilde, lda * lda * sizeof(double));
    }
    // printf("end \n");
}