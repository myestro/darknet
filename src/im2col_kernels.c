#include <string.h>

#include "im2col.h"
#include "cuda.h"
#include "im2col_kernels.cl"

#ifdef OPENCL

cl_kernel opencl_im2col_gpu_kernel;
cl_program opencl_im2col_kernels_program;

void im2col_kernel_init(void)
{
    opencl_load_buffer(im2col_kernel_source, strlen(im2col_kernel_source), &opencl_im2col_kernels_program);
    opencl_create_kernel(&opencl_im2col_kernels_program, "im2col_gpu_kernel", &opencl_im2col_gpu_kernel);
}

void im2col_kernel_release(void)
{
    clReleaseKernel(opencl_im2col_gpu_kernel); opencl_im2col_gpu_kernel = 0;
    clReleaseProgram(opencl_im2col_kernels_program); opencl_im2col_kernels_program = 0;
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

void im2col_ongpu(cl_mem im, int offset,
         int channels, int height, int width,
         int ksize, int stride, int pad, cl_mem data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    //int num_kernels = channels * height_col * width_col;

    dim3 dimGrid, dimBlock;
    dimGrid = dim3_create(height * width * channels, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    int zero = 0;

    opencl_kernel(opencl_im2col_gpu_kernel, dimGrid, dimBlock, 30,
        &im, sizeof(cl_mem),
        &offset,sizeof(cl_int),
        &channels, sizeof(cl_int),
        &height, sizeof(cl_int),
        &width, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &ksize, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &pad, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &height_col, sizeof(cl_int),
        &width_col, sizeof(cl_int),
        &data_col, sizeof(cl_mem),
        &zero, sizeof(cl_int));
}

#endif // OPENCL