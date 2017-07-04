#ifndef GEMM_H
#define GEMM_H

#ifdef GPU
#include "cuda.h"
#endif

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCL
void gemm_kernel_init(void);
void gemm_kernel_release(void);
#endif

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        GPU_DATA A_gpu, int offset_A, int lda, 
        GPU_DATA B_gpu, int offset_B, int ldb,
        float BETA,
        GPU_DATA C_gpu, int offset_C, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int offset_A, int lda, 
        float *B, int offset_B, int ldb,
        float BETA,
        float *C, int offset_C, int ldc);
#ifdef __cplusplus
}
#endif

#endif
#endif
