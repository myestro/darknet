#ifndef COL2IM_H
#define COL2IM_H

#ifdef GPU
#include "cuda.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU

#ifdef OPENCL
void col2im_kernel_init(void);
void col2im_kernel_release(void);
#endif

void col2im_ongpu(GPU_DATA data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, GPU_DATA data_im, int offset);
#endif

#ifdef __cplusplus
}
#endif

#endif
