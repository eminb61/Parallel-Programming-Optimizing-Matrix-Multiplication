const char* dgemm_desc = "Simple blocked dgemm.";

#define BLK_J 96
#define BLK_K 96
#define BLK_I 48

#define KS_LCD 24
#define KS_A 8
#define KS_B 6

#define ALIGNMENT 64
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define AA(i, j) A[i + lda * j]   // map alpha( i,j ) to array A
#define BB(i, j) B[i + lda * j]   // map beta( i,j ) to array B
#define CC(i, j) C[i + lda * j]   // map gamma( i,j ) to array C

#include<immintrin.h>
#include<string.h>
#include<stdio.h>



static inline void kernel8by8_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
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

static inline void kernel8by6_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
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

    MP_A += 8;
    MP_B += 6;
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
}

static inline void kernel12by4_packed(int lda, int K, double *MP_A, double *MP_B, double *C)
{
    __m256d gamma_0_0 = _mm256_loadu_pd( &CC( 0,0 ) );
    __m256d gamma_0_1 = _mm256_loadu_pd( &CC( 4,0 ) );
    __m256d gamma_0_2 = _mm256_loadu_pd( &CC( 8,0 ) );
    __m256d gamma_1_0 = _mm256_loadu_pd( &CC( 0,1 ) );
    __m256d gamma_1_1 = _mm256_loadu_pd( &CC( 4,1 ) );
    __m256d gamma_1_2 = _mm256_loadu_pd( &CC( 8,1 ) );
    __m256d gamma_2_0 = _mm256_loadu_pd( &CC( 0,2 ) );
    __m256d gamma_2_1 = _mm256_loadu_pd( &CC( 4,2 ) );
    __m256d gamma_2_2 = _mm256_loadu_pd( &CC( 8,2 ) );
    __m256d gamma_3_0 = _mm256_loadu_pd( &CC( 0,3 ) );
    __m256d gamma_3_1 = _mm256_loadu_pd( &CC( 4,3 ) );
    __m256d gamma_3_2 = _mm256_loadu_pd( &CC( 8,3 ) );

    __m256d beta_p_j;

    for ( int p=0; p<K; p++ ){

    __m256d alpha_p_0 = _mm256_loadu_pd( MP_A );
    __m256d alpha_p_1 = _mm256_loadu_pd( MP_A + 4 );
    __m256d alpha_p_2 = _mm256_loadu_pd( MP_A + 8 );

    beta_p_j = _mm256_broadcast_sd( MP_B );
    gamma_0_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_0_0 );
    gamma_0_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_0_1 );
    gamma_0_2 = _mm256_fmadd_pd( alpha_p_2, beta_p_j, gamma_0_2 );

    beta_p_j = _mm256_broadcast_sd( MP_B+1 );
    gamma_1_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_1_0 );
    gamma_1_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_1_1 );
    gamma_1_2 = _mm256_fmadd_pd( alpha_p_2, beta_p_j, gamma_1_2 );

    beta_p_j = _mm256_broadcast_sd( MP_B+2 );
    gamma_2_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_2_0 );
    gamma_2_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_2_1 );
    gamma_2_2 = _mm256_fmadd_pd( alpha_p_2, beta_p_j, gamma_2_2 );

    beta_p_j = _mm256_broadcast_sd( MP_B+3 );
    gamma_3_0 = _mm256_fmadd_pd( alpha_p_0, beta_p_j, gamma_3_0 );
    gamma_3_1 = _mm256_fmadd_pd( alpha_p_1, beta_p_j, gamma_3_1 );
    gamma_3_2 = _mm256_fmadd_pd( alpha_p_2, beta_p_j, gamma_3_2 );

    MP_A += 12;
    MP_B += 4;
    }
    /* Store the updated results.  This should be done more carefully since
        there may be an incomplete micro-tile. */
    _mm256_storeu_pd( &CC(0,0), gamma_0_0 );
    _mm256_storeu_pd( &CC(4,0), gamma_0_1 );
    _mm256_storeu_pd( &CC(8,0), gamma_0_2 );
    _mm256_storeu_pd( &CC(0,1), gamma_1_0 );
    _mm256_storeu_pd( &CC(4,1), gamma_1_1 );
    _mm256_storeu_pd( &CC(8,1), gamma_1_2 );
    _mm256_storeu_pd( &CC(0,2), gamma_2_0 );
    _mm256_storeu_pd( &CC(4,2), gamma_2_1 );
    _mm256_storeu_pd( &CC(8,2), gamma_2_2 );
    _mm256_storeu_pd( &CC(0,3), gamma_3_0 );
    _mm256_storeu_pd( &CC(4,3), gamma_3_1 );
    _mm256_storeu_pd( &CC(8,3), gamma_3_2 );
}



static inline void PackMicroPanelA_MRxKC_Pad(int K, int Medge, int lda, double *A, double *Atilde ){
/* Pack a micro-panel of A into buffer pointed to by Atilde. 
   This is an unoptimized implementation for general MR and KC. */
  /* March through A in column-major order, packing into Atilde as we go. */
    /* Full row size micro-panel.*/
    int Mpad = KS_A - Medge;
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
    int Medge = M % KS_A;
    int Mmax = M - Medge;
    int blk_num = Mmax / KS_A;
    int ptr_incr = (K - 1) * KS_A;
    int ptr_dcr = (K*blk_num - 1) * KS_A;
    for (int p=0; p<K; p++) {
        for (int i=0; i<Mmax; i+= KS_A) {
            /*unroll this*/
            *Atilde++ = AA(i, p);
            *Atilde++ = AA((i+1), p);
            *Atilde++ = AA((i+2), p);
            *Atilde++ = AA((i+3), p);
            *Atilde++ = AA((i+4), p);
            *Atilde++ = AA((i+5), p);
            *Atilde++ = AA((i+6), p);
            *Atilde++ = AA((i+7), p);
            // *Atilde++ = AA((i+8), p);
            // *Atilde++ = AA((i+9), p);
            // *Atilde++ = AA((i+10), p);
            // *Atilde++ = AA((i+11), p);
            Atilde += ptr_incr;
        }
        Atilde -= ptr_dcr;
    }
    Atilde += (blk_num - 1) * K * KS_A;
    if (Medge != 0){
        PackMicroPanelA_MRxKC_Pad(K, Medge, lda, &AA(Mmax, 0), Atilde);
    }
}

static inline void PackMicroPanelB_KCxNR_Pad(int K, int Nedge, int lda, double *B, double *Btilde){
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR.
   k is assumed to be less then or equal to KC.
   n is assumed to be less then or equal to NR.  */
    int Npad = KS_B - Nedge;
    for (int p=0; p<K; p++){
        for (int j=0; j<Nedge; j++){
            *Btilde++ = BB(p, j);
        }
        for (int i=0; i<Npad; i++){
            *Btilde++ = 0;
        }
    }
}

static inline void PackPanelB_KCxNC(int K, int N, int lda, double *B, double *Btilde){
/* Pack a k x n panel of B in to a KC x NC buffer.
   The block is copied into Btilde a micro-panel at a time. */
    int Nedge = N % KS_B;
    int Nmax = N - Nedge;
    for (int j=0; j<Nmax; j+= KS_B){
        // PackMicroPanelB_KCxNR(K, lda, &BB(0, j), Btilde);
        // Btilde += K * KS_B;
        // double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
        // b_ptr0 = &BB(0, j);
        // b_ptr1 = b_ptr0 + lda;
        // b_ptr2 = b_ptr1 + lda;
        // b_ptr3 = b_ptr2 + lda;
        // for (int p=0; p<K; p++){
        //     *Btilde++ = *b_ptr0++;
        //     *Btilde++ = *b_ptr1++;
        //     *Btilde++ = *b_ptr2++;
        //     *Btilde++ = *b_ptr3++;
        //     // for ( int j=0; j<KS_B; j++ ){
        //     //     *Btilde++ = BB(p, j);
        //     // }  
        // }
        double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3, *b_ptr4, *b_ptr5;
        b_ptr0 = &BB(0, j);
        b_ptr1 = b_ptr0 + lda;
        b_ptr2 = b_ptr1 + lda;
        b_ptr3 = b_ptr2 + lda;
        b_ptr4 = b_ptr3 + lda;
        b_ptr5 = b_ptr4 + lda;
        for (int p=0; p<K; p++){
            *Btilde++ = *b_ptr0++;
            *Btilde++ = *b_ptr1++;
            *Btilde++ = *b_ptr2++;
            *Btilde++ = *b_ptr3++;
            *Btilde++ = *b_ptr4++;
            *Btilde++ = *b_ptr5++;
            // for ( int j=0; j<KS_B; j++ ){
            //     *Btilde++ = BB(p, j);
            // }  
        }
        // double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3, *b_ptr4, *b_ptr5, *b_ptr6, *b_ptr7;
        // b_ptr0 = &BB(0, j);
        // b_ptr1 = b_ptr0 + lda;
        // b_ptr2 = b_ptr1 + lda;
        // b_ptr3 = b_ptr2 + lda;
        // b_ptr4 = b_ptr3 + lda;
        // b_ptr5 = b_ptr4 + lda;
        // b_ptr6 = b_ptr5 + lda;
        // b_ptr7 = b_ptr6 + lda;
        // for (int p=0; p<K; p++){
        //     *Btilde++ = *b_ptr0++;
        //     *Btilde++ = *b_ptr1++;
        //     *Btilde++ = *b_ptr2++;
        //     *Btilde++ = *b_ptr3++;
        //     *Btilde++ = *b_ptr4++;
        //     *Btilde++ = *b_ptr5++;
        //     *Btilde++ = *b_ptr6++;
        //     *Btilde++ = *b_ptr7++;
        //     // for ( int j=0; j<KS_B; j++ ){
        //     //     *Btilde++ = BB(p, j);
        //     // }  
        // }
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
    int blk = (lda + KS_A - 1) / KS_A;
    int ldc = blk * KS_A;
    // int blk_j = min(BLK_J, ldc);
    // int blk_k = min(BLK_K, ldc);
    // int blk_i = min(BLK_I, ldc);
    int blk_j = BLK_J;
    int blk_k = BLK_K;
    int blk_i = BLK_I;
    // int blk_j = 96;
    // int blk_i = 96;
    // int blk_k = 96;
    // if (lda < 96) {
    //     blk_k = 48;
    //     blk_i = 48;
    //     blk_k = 48;
    // }
    Ctilde = _mm_malloc(ldc * ldc * sizeof(double), ALIGNMENT);
    if (ldc != lda) {
        for (int i = 0; i < lda; i ++){
            memcpy(Ctilde + ldc * i, C + lda * i, lda * sizeof(double));
        }
    }else{
        memcpy(Ctilde, C, lda * lda * sizeof(double));
    }
    // double *Btilde = (double *) malloc(BLK_K * BLK_J * sizeof(double)); // Target L3, KC * NC
    Btilde = _mm_malloc(blk_k * blk_j * sizeof(double), ALIGNMENT);
    // double *Atilde = (double *) malloc(BLK_I * BLK_K * sizeof(double)); // Target L2, MC * KC
    Atilde = _mm_malloc(blk_i * blk_k * sizeof(double), ALIGNMENT);
    for (int j = 0; j < lda; j += blk_j) {
        // For each block-column of B
        int N = min(blk_j, lda - j);
        for (int k = 0; k < lda; k += blk_k) {
            // Accumulate block dgemms into block of C
            int K = min(blk_k, lda - k);
            PackPanelB_KCxNC(K, N, lda, &BB(k, j), Btilde);
            for (int i = 0; i < lda; i += blk_i) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(blk_i, lda - i);
                PackBlockA_MCxKC(M, K, lda, &AA(i, k), Atilde);
                // Perform individual block dgemm
                c_p = &Ctilde[i + j*ldc];
                for (int v = 0; v < N; v+=KS_B) {
                    for (int u = 0; u < M; u+=KS_A) {
                        // kernel4by4_packed(ldc, K, &Atilde[u*K], b_ptr, &c_p[u + v*ldc]);
                        kernel8by6_packed(ldc, K, &Atilde[u*K], &Btilde[v*K], &c_p[u + v*ldc]);
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