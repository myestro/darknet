#include <string.h>

#include "col2im.h"
#include "cuda.h"
#include "col2im_kernels.cl"

#ifdef OPENCL

cl_program opencl_col2im_program;
cl_kernel opencl_col2im_kernel;

void col2im_kernel_init(void)
{
    opencl_load_buffer(col2im_kernel_source, strlen(col2im_kernel_source), &opencl_col2im_program);
    opencl_create_kernel(&opencl_col2im_program, "col2im_gpu_kernel",
        &opencl_col2im_kernel);
}

void col2im_kernel_release(void)
{
    clReleaseKernel(opencl_col2im_kernel);
    clReleaseProgram(opencl_col2im_program);

    opencl_col2im_kernel = 0;
    opencl_col2im_program = 0;
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

void col2im_ongpu(cl_mem data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im, int offset){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;

    dim3 dimGrid, dimBlock;
    dimGrid = dim3_create(height * width * channels, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    int zero = 0;

    opencl_kernel(opencl_col2im_kernel, dimGrid, dimBlock, 32,
        &num_kernels, sizeof(cl_int),
        &data_col, sizeof(cl_mem),
        &zero, sizeof(cl_int),
        &height, sizeof(cl_int),
        &width, sizeof(cl_int),
        &channels, sizeof(cl_int), 
        &ksize, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &height_col, sizeof(cl_int),
        &width_col, sizeof(cl_int),
        &data_im, sizeof(cl_mem),
        &offset, sizeof(cl_int));
}

#endif //OPENCL