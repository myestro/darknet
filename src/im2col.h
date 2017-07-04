#ifndef IM2COL_H
#define IM2COL_H

#ifdef GPU
#include "cuda.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

#ifdef OPENCL
void im2col_kernel_init(void);
void im2col_kernel_release(void);
#endif

void im2col_ongpu(GPU_DATA im, int offset,
         int channels, int height, int width,
         int ksize, int stride, int pad, GPU_DATA data_col);

#endif


#ifdef __cplusplus
}
#endif

#endif
