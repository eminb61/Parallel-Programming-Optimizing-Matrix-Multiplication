const char* dgemm_desc = "Simple blocked dgemm.";

// #define BLOCK_SIZE 64
// #define BLOCK_SIZE_L1 64
// #define BLOCK_SIZE_L2 256
#define BLK_J 512
#define BLK_K 128
#define BLK_I 64

#define KS 8

#define ALIGNMENT 64
#define min(a, b) (((a) < (b)) ? (a) : (b))

// #define alpha( i,j ) A[ (j)*lda + (i) ]   // map alpha( i,j ) to array A
// #define beta( i,j )  B[ (j)*lda + (i) ]   // map beta( i,j ) to array B
// #define gamma( i,j ) C[ (j)*lda + (i) ]   // map gamma( i,j ) to array C

#define AA(i, j) A[i + lda * j]   // map alpha( i,j ) to array A
#define BB(i, j) B[i + lda * j]   // map beta( i,j ) to array B
#define CC(i, j) C[i + lda * j]   // map gamma( i,j ) to array C

#include<immintrin.h>
#include<string.h>

// void pointerFuncA(double* iptr){
//     /*Print the value pointed to by iptr*/
//     printf("A \n");
//     printf("First Element:  %f\n", iptr[0]);
//     printf("991:  %f\n", iptr[990]);
//     printf("992:  %f\n", iptr[991]);
//     printf("Last Element:  %f\n", iptr[1023]);

//     /*Print the address pointed to by iptr*/
//     printf("Address of value: %p\n", (void*)iptr);

//     /*Print the address of iptr itself*/
//     // printf("Address of iptr: %p\n", (void*)&iptr);
// }
// void pointerFuncB(double* iptr){
//     /*Print the value pointed to by iptr*/
//     printf("B \n");
//     printf("First Element:  %f\n", iptr[0]);
//     printf("991:  %f\n", iptr[990]);
//     printf("992:  %f\n", iptr[991]);
//     printf("Last Element:  %f\n", iptr[1023]);

//     /*Print the address pointed to by iptr*/
//     printf("Address of value: %p\n", (void*)iptr);

//     /*Print the address of iptr itself*/
//     // printf("Address of iptr: %p\n", (void*)&iptr);
// }

// static void kernel4by4_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
// {
//   __m256d gamma_0123_0 = _mm256_loadu_pd( &CC( 0,0 ) );
//   __m256d gamma_0123_1 = _mm256_loadu_pd( &CC( 0,1 ) );
//   __m256d gamma_0123_2 = _mm256_loadu_pd( &CC( 0,2 ) );
//   __m256d gamma_0123_3 = _mm256_loadu_pd( &CC( 0,3 ) );

//   __m256d beta_p_j;
   	
//   for ( int p=0; p<K; p++ ){
//     /* load alpha( 0:3, p ) */
//     __m256d alpha_0123_p = _mm256_loadu_pd( MP_A );

//     /* load beta( p, 0 ); update gamma( 0:3, 0 ) */
//     beta_p_j = _mm256_broadcast_sd( MP_B );
//     gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );

//     /* load beta( p, 1 ); update gamma( 0:3, 1 ) */
//     beta_p_j = _mm256_broadcast_sd( MP_B+1 );
//     gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );

//     /* load beta( p, 2 ); update gamma( 0:3, 2 ) */
//     beta_p_j = _mm256_broadcast_sd( MP_B+2 );
//     gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );

//     /* load beta( p, 3 ); update gamma( 0:3, 3 ) */
//     beta_p_j = _mm256_broadcast_sd( MP_B+3 );
//     gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );

//     MP_A += 4;
//     MP_B += 4;
//   }

//   /* Store the updated results.  This should be done more carefully since
//      there may be an incomplete micro-tile. */
//   _mm256_storeu_pd( &CC(0,0), gamma_0123_0 );
//   _mm256_storeu_pd( &CC(0,1), gamma_0123_1 );
//   _mm256_storeu_pd( &CC(0,2), gamma_0123_2 );
//   _mm256_storeu_pd( &CC(0,3), gamma_0123_3 );
// }

static void kernel8by8_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
{
  __m256d gamma_0_0 = _mm256_loadu_pd( &CC( 0,0 ) );
  __m256d gamma_0_1 = _mm256_loadu_pd( &CC( 4,0 ) );
  __m256d gamma_1_0 = _mm256_loadu_pd( &CC( 0,1 ) );
  __m256d gamma_1_1 = _mm256_loadu_pd( &CC( 4,1 ) );
  __m256d gamma_2_0 = _mm256_loadu_pd( &CC( 0,2 ) );
  __m256d gamma_2_1 = _mm256_loadu_pd( &CC( 4,2 ) );
  __m256d gamma_3_0 = _mm256_loadu_pd( &CC( 0,3 ) );
  __m256d gamma_3_1 = _mm256_loadu_pd( &CC( 4,3 ) );
  __m256d gamma_4_0 = _mm256_loadu_pd( &CC( 0,4 ) );
  __m256d gamma_4_1 = _mm256_loadu_pd( &CC( 4,4 ) );
  __m256d gamma_5_0 = _mm256_loadu_pd( &CC( 0,5 ) );
  __m256d gamma_5_1 = _mm256_loadu_pd( &CC( 4,5 ) );
  __m256d gamma_6_0 = _mm256_loadu_pd( &CC( 0,6 ) );
  __m256d gamma_6_1 = _mm256_loadu_pd( &CC( 4,6 ) );
  __m256d gamma_7_0 = _mm256_loadu_pd( &CC( 0,7 ) );
  __m256d gamma_7_1 = _mm256_loadu_pd( &CC( 4,7 ) );


  __m256d beta_p_j;
   	
  for ( int p=0; p<K; p++ ){
    
    __m256d alpha_p_0 = _mm256_loadu_pd( MP_A );
    __m256d alpha_p_1 = _mm256_loadu_pd( MP_A + 4 );

    beta_p_j = _mm256_broadcast_sd( MP_B );
    gamma_0_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_0_0 );
    gamma_0_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_0_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+1 );
    gamma_1_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_1_0 );
    gamma_1_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_1_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+2 );
    gamma_2_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_2_0 );
    gamma_2_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_2_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+3 );
    gamma_3_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_3_0 );
    gamma_3_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_3_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+4 );
    gamma_4_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_4_0 );
    gamma_4_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_4_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+5 );
    gamma_5_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_5_0 );
    gamma_5_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_5_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+6 );
    gamma_6_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_6_0 );
    gamma_6_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_6_1 );

    beta_p_j = _mm256_broadcast_sd( MP_B+7 );
    gamma_7_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_7_0 );
    gamma_7_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_7_1 );

    MP_A += 8;
    MP_B += 8;
  }

  /* Store the updated results.  This should be done more carefully since
     there may be an incomplete micro-tile. */
  _mm256_storeu_pd( &CC(0,0), gamma_0_0 );
  _mm256_storeu_pd( &CC(4,0), gamma_0_1 );
  _mm256_storeu_pd( &CC(0,1), gamma_1_0 );
  _mm256_storeu_pd( &CC(4,1), gamma_1_1 );
  _mm256_storeu_pd( &CC(0,2), gamma_2_0 );
  _mm256_storeu_pd( &CC(4,2), gamma_2_1 );
  _mm256_storeu_pd( &CC(0,3), gamma_3_0 );
  _mm256_storeu_pd( &CC(4,3), gamma_3_1 );
  _mm256_storeu_pd( &CC(0,4), gamma_4_0 );
  _mm256_storeu_pd( &CC(4,4), gamma_4_1 );
  _mm256_storeu_pd( &CC(0,5), gamma_5_0 );
  _mm256_storeu_pd( &CC(4,5), gamma_5_1 );
  _mm256_storeu_pd( &CC(0,6), gamma_6_0 );
  _mm256_storeu_pd( &CC(4,6), gamma_6_1 );
  _mm256_storeu_pd( &CC(0,7), gamma_7_0 );
  _mm256_storeu_pd( &CC(4,7), gamma_7_1 );
}

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

static inline void copy_mem (int K, double* a_src, double* a_dest) {
  for (int i = 0; i < K; ++i) 
  { 
    /* Unroll */
    // *a_dest++ = *a_src;
    // *a_dest++ = *(a_src+1);
    // *a_dest++ = *(a_src+2);
    // *a_dest++ = *(a_src+3);
    // a_src += 4;
    for (int j = 0; j < KS; j++){
        *a_dest++ = *a_src++;
    }
  }
}

static inline void PackMicroPanelA_MRxKC(int K, int lda, double *A, double *Atilde ){
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
  /* March through A in column-major order, packing into Atilde as we go. */
    /* Full row size micro-panel.*/
    for (int p=0; p<K; p++){
        /*unroll this*/
        // *Atilde++ = AA(0, p);
        // *Atilde++ = AA(1, p);
        // *Atilde++ = AA(2, p);
        // *Atilde++ = AA(3, p);
        for ( int i=0; i<KS; i++ ){
            *Atilde++ = AA(i, p);
        }  
    }
}

static inline void PackMicroPanelA_MRxKC_Pad(int K, int Medge, int lda, double *A, double *Atilde ){
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
  /* March through A in column-major order, packing into Atilde as we go. */
    /* Full row size micro-panel.*/
    int Mpad = KS - Medge;
    for (int p=0; p<K; p++){
        for (int i=0; i<Medge; i++) {
            *Atilde++ = AA(i, p);
        }
        for (int i=0; i<Mpad; i++){
            *Atilde++ = 0;
        }
    }
}

static inline void PackBlockA_MCxKC(int M, int K, int lda, double *A, double *Atilde ){
/* Pack a  m x k block of A into a MC x KC buffer.   MC is assumed to
    be a multiple of MR.  The block is packed into Atilde a micro-panel
    at a time. If necessary, the last micro-panel is padded with rows
    of zeroes. */
    int Medge = M % KS;
    int Mmax = M - Medge;
    for (int i=0; i<Mmax; i+= KS){
        PackMicroPanelA_MRxKC(K, lda, &AA(i, 0), Atilde);
        Atilde += K * KS;
    }
    if (Medge != 0){
        PackMicroPanelA_MRxKC_Pad(K, Medge, lda, &AA(Mmax, 0), Atilde);
    }
}

static inline void PackMicroPanelB_KCxNR_Pad(int K, int Nedge, int lda, double *B, double *Btilde){
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
    int Npad = KS - Nedge;
    for (int p=0; p<K; p++){
        for (int j=0; j<Nedge; j++){
            *Btilde++ = BB(p, j);
        }
        for (int i=0; i<Npad; i++){
            *Btilde++ = 0;
        }
    }
}

static inline void PackMicroPanelB_KCxNR(int K, int lda, double *B, double *Btilde){
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
    for (int p=0; p<K; p++){
        // *Btilde++ = BB(p, 0);
        // *Btilde++ = BB(p, 1);
        // *Btilde++ = BB(p, 2);
        // *Btilde++ = BB(p, 3);
        for ( int j=0; j<KS; j++ ){
            *Btilde++ = BB(p, j);
        }  
    }
}

static inline void PackPanelB_KCxNC(int K, int N, int lda, double *B, double *Btilde){
/* Pack a k x n panel of B in to a KC x NC buffer.
   The block is copied into Btilde a micro-panel at a time. */
    int Nedge = N % KS;
    int Nmax = N - Nedge;
    for (int j=0; j<Nmax; j+= KS){
        PackMicroPanelB_KCxNR(K, lda, &BB(0, j), Btilde);
        Btilde += K * KS;
    }
    if (Nedge != 0){
        PackMicroPanelB_KCxNR_Pad(K, Nedge, lda, &BB(0, Nmax), Btilde);
    }
   
}
static double * restrict Ctilde __attribute__ ((aligned(ALIGNMENT)));
static double * restrict Atilde __attribute__ ((aligned(ALIGNMENT)));
static double * restrict Btilde __attribute__ ((aligned(ALIGNMENT)));

void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    double *c_p;
    int blk = (lda + KS- 1) / KS;
    int ldc = blk * KS;
    Ctilde = _mm_malloc(ldc * ldc * sizeof(double), ALIGNMENT);
    if (ldc != lda) {
        for (int i = 0; i < lda; i ++){
            memcpy(Ctilde + ldc * i, C + lda * i, lda * sizeof(double));
        }
    }else{
        memcpy(Ctilde, C, lda * lda * sizeof(double));
    }
    // double *Btilde = (double *) malloc(BLK_K * BLK_J * sizeof(double)); // Target L3, KC * NC
    Btilde = _mm_malloc(BLK_K * BLK_J * sizeof(double), ALIGNMENT);
    // double *Atilde = (double *) malloc(BLK_I * BLK_K * sizeof(double)); // Target L2, MC * KC
    Atilde = _mm_malloc(BLK_I * BLK_K * sizeof(double), ALIGNMENT);
    for (int j = 0; j < lda; j += BLK_J) {
        // For each block-column of B
        int N = min(BLK_J, lda - j);
        
        
        for (int k = 0; k < lda; k += BLK_K) {
            // Accumulate block dgemms into block of C
            int K = min(BLK_K, lda - k);
            
            PackPanelB_KCxNC(K, N, lda, &BB(k, j), Btilde);
            for (int i = 0; i < lda; i += BLK_I) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLK_I, lda - i);
                PackBlockA_MCxKC(M, K, lda, &AA(i, k), Atilde);
                // Perform individual block dgemm
                c_p = &Ctilde[i + j*ldc];
                double B_block[KS*K];
                double *b_ptr;
                for (int v = 0; v < N; v+=KS) {
                    b_ptr = &B_block[0];
                    copy_mem(K, &Btilde[v*K], b_ptr);
                    for (int u = 0; u < M; u+=KS) {
                        // a_ptr = &A_block[0];
                        // if (v == 0) copy_mem(K, &Atilde[u*K], a_ptr);
                        // kernel4by4_packed(ldc, K, &Atilde[u*K], b_ptr, &c_p[u + v*ldc]);
                        kernel8by8_packed(ldc, K, &Atilde[u*K], b_ptr, &c_p[u + v*ldc]);
                    }
                }
            }
        }
    }
    if (ldc != lda) {
        for (int i = 0; i < lda; i ++){
            memcpy(C + lda * i, Ctilde + ldc * i, ldc * sizeof(double));
        }
    }else{
        memcpy(C, Ctilde, lda * lda * sizeof(double));
    }
    // free(Atilde);
    // free(Btilde);
}