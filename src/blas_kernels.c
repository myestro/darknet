#include <assert.h>
#include <string.h>

#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include "blas_kernels.cl"

#ifdef OPENCL

cl_program opencl_blas_kernel_program = 0;
cl_kernel opencl_scale_bias_kernel = 0;
cl_kernel opencl_scale_bias_kernel2 = 0;
cl_kernel opencl_backward_scale_kernel = 0;
cl_kernel opencl_add_bias_kernel = 0;
cl_kernel opencl_add_bias_kernel2 = 0;
cl_kernel opencl_backward_bias_kernel = 0;
cl_kernel opencl_adam_kernel = 0;
cl_kernel opencl_normalize_kernel = 0;
cl_kernel opencl_normalize_delta_kernel = 0;
cl_kernel opencl_variance_delta_kernel = 0;
cl_kernel opencl_accumulate_kernel = 0;
cl_kernel opencl_fast_mean_delta_kernel = 0;
cl_kernel opencl_fast_variance_delta_kernel = 0;
cl_kernel opencl_mean_delta_kernel = 0;
cl_kernel opencl_mean_kernel = 0;
cl_kernel opencl_variance_kernel = 0;
cl_kernel opencl_reorg_kernel = 0;
cl_kernel opencl_axpy_kernel = 0;
cl_kernel opencl_pow_kernel = 0;
cl_kernel opencl_const_kernel = 0;
cl_kernel opencl_constrain_kernel = 0;
cl_kernel opencl_supp_kernel = 0;
cl_kernel opencl_add_kernel = 0;
cl_kernel opencl_scal_kernel = 0;
cl_kernel opencl_fill_kernel = 0;
cl_kernel opencl_mask_kernel = 0;
cl_kernel opencl_copy_kernel = 0;
cl_kernel opencl_mul_kernel = 0;
cl_kernel opencl_fast_mean_kernel = 0;
cl_kernel opencl_fast_variance_kernel = 0;
cl_kernel opencl_flatten_kernel = 0;
cl_kernel opencl_shortcut_kernel = 0;
cl_kernel opencl_smooth_l1_kernel = 0;
cl_kernel opencl_l2_kernel = 0;
cl_kernel opencl_l1_kernel = 0;
cl_kernel opencl_weighted_sum_kernel = 0;
cl_kernel opencl_weighted_delta_kernel = 0;
cl_kernel opencl_mult_add_into_kernel = 0;
cl_kernel opencl_softmax_kernel = 0;

void blas_kernel_init(void)
{
    opencl_load_buffer(blas_kernel_source, strlen(blas_kernel_source), &opencl_blas_kernel_program);
    
    opencl_create_kernel(&opencl_blas_kernel_program, "scale_bias_kernel", &opencl_scale_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "scale_bias_kernel2", &opencl_scale_bias_kernel2);
    opencl_create_kernel(&opencl_blas_kernel_program, "backward_scale_kernel", &opencl_backward_scale_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "add_bias_kernel", &opencl_add_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "add_bias_kernel2", &opencl_add_bias_kernel2);
    opencl_create_kernel(&opencl_blas_kernel_program, "backward_bias_kernel", &opencl_backward_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "adam_kernel", &opencl_adam_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "normalize_kernel", &opencl_normalize_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "normalize_delta_kernel", &opencl_normalize_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "variance_delta_kernel", &opencl_variance_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "accumulate_kernel", &opencl_accumulate_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fast_mean_delta_kernel", &opencl_fast_mean_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fast_variance_delta_kernel", &opencl_fast_variance_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mean_delta_kernel", &opencl_mean_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mean_kernel", &opencl_mean_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "variance_kernel", &opencl_variance_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "reorg_kernel", &opencl_reorg_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "axpy_kernel", &opencl_axpy_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "pow_kernel", &opencl_pow_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "const_kernel", &opencl_const_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "constrain_kernel", &opencl_constrain_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "supp_kernel", &opencl_supp_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "add_kernel", &opencl_add_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "scal_kernel", &opencl_scal_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fill_kernel", &opencl_fill_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mask_kernel", &opencl_mask_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "copy_kernel", &opencl_copy_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mul_kernel", &opencl_mul_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fast_mean_kernel", &opencl_fast_mean_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fast_variance_kernel", &opencl_fast_variance_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "flatten_kernel", &opencl_flatten_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "shortcut_kernel", &opencl_shortcut_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "smooth_l1_kernel", &opencl_smooth_l1_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "l2_kernel", &opencl_l2_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "l1_kernel", &opencl_l1_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "weighted_sum_kernel", &opencl_weighted_sum_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "weighted_delta_kernel", &opencl_weighted_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mult_add_into_kernel", &opencl_mult_add_into_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "softmax_kernel", &opencl_softmax_kernel);
}

void blas_kernel_release(void)
{
    clReleaseKernel(opencl_scale_bias_kernel); opencl_scale_bias_kernel = 0;
    clReleaseKernel(opencl_scale_bias_kernel2); opencl_scale_bias_kernel2 = 0;
    clReleaseKernel(opencl_backward_scale_kernel); opencl_backward_scale_kernel = 0;
    clReleaseKernel(opencl_add_bias_kernel); opencl_add_bias_kernel = 0;
    clReleaseKernel(opencl_add_bias_kernel2); opencl_add_bias_kernel2 = 0;
    clReleaseKernel(opencl_backward_bias_kernel); opencl_backward_bias_kernel = 0;
    clReleaseKernel(opencl_adam_kernel); opencl_adam_kernel = 0;
    clReleaseKernel(opencl_normalize_kernel); opencl_normalize_kernel = 0;
    clReleaseKernel(opencl_normalize_delta_kernel); opencl_normalize_delta_kernel = 0;
    clReleaseKernel(opencl_variance_delta_kernel); opencl_variance_delta_kernel = 0;
    clReleaseKernel(opencl_accumulate_kernel); opencl_accumulate_kernel = 0;
    clReleaseKernel(opencl_fast_mean_delta_kernel); opencl_fast_mean_delta_kernel = 0;
    clReleaseKernel(opencl_fast_variance_delta_kernel); opencl_fast_variance_delta_kernel = 0;
    clReleaseKernel(opencl_mean_delta_kernel); opencl_mean_delta_kernel = 0;
    clReleaseKernel(opencl_mean_kernel); opencl_mean_kernel = 0;
    clReleaseKernel(opencl_variance_kernel); opencl_variance_kernel = 0;
    clReleaseKernel(opencl_reorg_kernel); opencl_reorg_kernel = 0;
    clReleaseKernel(opencl_axpy_kernel); opencl_axpy_kernel = 0;
    clReleaseKernel(opencl_pow_kernel); opencl_pow_kernel = 0;
    clReleaseKernel(opencl_const_kernel); opencl_const_kernel = 0;
    clReleaseKernel(opencl_constrain_kernel); opencl_constrain_kernel = 0;
    clReleaseKernel(opencl_supp_kernel); opencl_supp_kernel = 0;
    clReleaseKernel(opencl_add_kernel); opencl_add_kernel = 0;
    clReleaseKernel(opencl_scal_kernel); opencl_scal_kernel = 0;
    clReleaseKernel(opencl_fill_kernel); opencl_fill_kernel = 0;
    clReleaseKernel(opencl_mask_kernel); opencl_mask_kernel = 0;
    clReleaseKernel(opencl_copy_kernel); opencl_copy_kernel = 0;
    clReleaseKernel(opencl_mul_kernel); opencl_mul_kernel = 0;
    clReleaseKernel(opencl_fast_mean_kernel); opencl_fast_mean_kernel = 0;
    clReleaseKernel(opencl_fast_variance_kernel); opencl_fast_variance_kernel = 0;
    clReleaseKernel(opencl_flatten_kernel); opencl_flatten_kernel = 0;
    clReleaseKernel(opencl_shortcut_kernel); opencl_shortcut_kernel = 0;
    clReleaseKernel(opencl_smooth_l1_kernel); opencl_smooth_l1_kernel = 0;
    clReleaseKernel(opencl_l2_kernel); opencl_l2_kernel = 0;
    clReleaseKernel(opencl_l1_kernel); opencl_l1_kernel = 0;
    clReleaseKernel(opencl_weighted_sum_kernel); opencl_weighted_sum_kernel = 0;
    clReleaseKernel(opencl_weighted_delta_kernel); opencl_weighted_delta_kernel = 0;
    clReleaseKernel(opencl_mult_add_into_kernel); opencl_mult_add_into_kernel = 0;
    clReleaseKernel(opencl_softmax_kernel); opencl_softmax_kernel = 0;

    clReleaseProgram(opencl_blas_kernel_program); opencl_blas_kernel_program = 0;
}

void scale_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
{
    int N = batch * n * size;
    dim3 dimGrid, dimBlock;
    dimGrid = dim3_create(N, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_scale_bias_kernel2, dimGrid, dimBlock, 12, &N, sizeof(cl_int), &output, sizeof(cl_mem), &biases, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}

//void scale_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
//{
//    dim3 dimGrid, dimBlock;
//    dimGrid = dim3_create((size-1)/BLOCK + 1, n ,batch);
//    dimBlock = dim3_create(BLOCK, 1, 1);
//
//    opencl_kernel(opencl_scale_bias_kernel, dimGrid, dimBlock, 8, &output, sizeof(cl_mem), &biases, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int));
//}


void backward_scale_gpu(cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates)
{
    dim3 dimN, dimBlock;
    dimN = dim3_create(n, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_backward_scale_kernel, dimN, dimBlock, 12, &x_norm, sizeof(cl_mem), &delta, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int), &scale_updates, sizeof(cl_mem));
}


void add_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
{
    int N = batch * n * size;
    dim3 dimGrid, dimBlock;
    dimGrid = dim3_create(N, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_add_bias_kernel2, dimGrid, dimBlock, 12, &N, sizeof(cl_int), &output, sizeof(cl_mem), &biases, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}

//void add_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
//{
//    dim3 dimGrid, dimBlock;
//    dimGrid = dim3_create((size-1)/BLOCK + 1, n, batch);
//    dimBlock = dim3_create(BLOCK, 1, 1);
//
//    opencl_kernel(opencl_add_bias_kernel, dimGrid, dimBlock, 8, &output, sizeof(cl_mem), &biases, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int));
//}


void backward_bias_gpu(cl_mem bias_updates, cl_mem delta, int batch, int n, int size)
{
    dim3 dimN, dimBlock;
    dimN = dim3_create(n, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_backward_bias_kernel, dimN, dimBlock, 10, &bias_updates, sizeof(cl_mem), &delta, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void adam_gpu(int n, cl_mem x, cl_mem m, cl_mem v, float B1, float B2, float rate, float eps, int t)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_adam_kernel, dimGrid, dimBlock, 18, &n, sizeof(cl_int), &x, sizeof(cl_mem), &m, sizeof(cl_mem), &v, sizeof(cl_mem), &B1, sizeof(cl_float), &B2, sizeof(cl_float), &rate, sizeof(cl_float), &eps, sizeof(cl_float), &t, sizeof(cl_int));
}


void normalize_delta_gpu(cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta)
{
    size_t N = batch*filters*spatial;
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_normalize_kernel, dimGrid, dimBlock, 20, &N, sizeof(cl_mem), &x, sizeof(cl_mem), &mean, sizeof(cl_mem), &variance, sizeof(cl_mem), &mean_delta, sizeof(cl_mem), &variance_delta, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &delta, sizeof(cl_mem));
}


void mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(filters);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_mean_delta_kernel, dimGrid, dimBlock, 12, &delta, sizeof(cl_mem), &variance, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean_delta, sizeof(cl_mem));
}

void fast_mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta)
{
    dim3 dimFilters, dimBlock;
    dimFilters = dim3_create(filters, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_fast_mean_delta_kernel, dimFilters, dimBlock, 12, &delta, sizeof(cl_mem), &variance, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean_delta, sizeof(cl_mem));
}

void fast_variance_delta_gpu(cl_mem x, cl_mem delta, cl_mem mean, cl_mem variance, int batch, int filters, int spatial, cl_mem variance_delta)
{
    dim3 dimFilters, dimBlock;
    dimFilters = dim3_create(filters, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_fast_variance_delta_kernel, dimFilters, dimBlock, 16, &x, sizeof(cl_mem), &delta, sizeof(cl_mem), &mean, sizeof(cl_mem), &variance, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance_delta, sizeof(cl_mem));
}


void normalize_gpu(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_normalize_kernel, dimN, dimBlock, 14, &N, sizeof(cl_int), &x, sizeof(cl_mem), &mean, sizeof(cl_mem), &variance, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int));
}


void fast_mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean)
{
    dim3 dimFilters, dimBlock;
    dimFilters = dim3_create(filters, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_fast_mean_kernel, dimFilters, dimBlock, 10, &x, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean, sizeof(cl_mem));
}

void fast_variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance)
{
    dim3 dimFilters, dimBlock;
    dimFilters = dim3_create(filters, 1, 1);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_fast_variance_kernel, dimFilters, dimBlock, 12, &x, sizeof(cl_mem), &mean, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance, sizeof(cl_mem));
}


void mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean)
{
    dim3 dimFilters, dimBlock;
    dimFilters = cuda_gridsize(filters);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_mean_kernel, dimFilters, dimBlock, 10, &x, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean, sizeof(cl_mem));
}


void variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance)
{
    dim3 dimFilters, dimBlock;
    dimFilters = cuda_gridsize(filters);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_variance_kernel, dimFilters, dimBlock, 12, &x, sizeof(cl_mem), &mean, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance, sizeof(cl_mem));
}


void axpy_ongpu(int N, float ALPHA, cl_mem  X, int INCX, cl_mem  Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}


void pow_ongpu(int N, float ALPHA, cl_mem  X, int INCX, cl_mem  Y, int INCY)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_pow_kernel, dimGrid, dimBlock, 12, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int), &Y, sizeof(cl_mem), &INCY, sizeof(cl_int));
}


void axpy_ongpu_offset(int N, float ALPHA, cl_mem  X, int OFFX, int INCX, cl_mem  Y, int OFFY, int INCY)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_axpy_kernel, dimGrid, dimBlock, 16, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}


void copy_ongpu(int N, cl_mem  X, int INCX, cl_mem  Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}


void mul_ongpu(int N, cl_mem  X, int INCX, cl_mem  Y, int INCY)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_mul_kernel, dimGrid, dimBlock, 10, &N, sizeof(cl_int), &X, sizeof(cl_mem), &INCX, sizeof(cl_int), &Y, sizeof(cl_mem), &INCY, sizeof(cl_int));
}


void copy_ongpu_offset(int N, cl_mem  X, int OFFX, int INCX, cl_mem  Y, int OFFY, int INCY)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_copy_kernel, dimGrid, dimBlock, 14, &N, sizeof(cl_int), &X, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}


void flatten_ongpu(cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out)
{
    int size = spatial*batch*layers;
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_flatten_kernel, dimGrid, dimBlock, 14, &size, sizeof(cl_int), &x, sizeof(cl_mem), &spatial, sizeof(cl_int), &layers, sizeof(cl_int), &batch, sizeof(cl_int), &forward, sizeof(cl_int), &out, sizeof(cl_mem));
}


void reorg_ongpu(cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out)
{
    int size = w*h*c*batch;
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_reorg_kernel, dimGrid, dimBlock, 18, &size, sizeof(cl_int), &x, sizeof(cl_mem), &w, sizeof(cl_int), &h, sizeof(cl_int), &c, sizeof(cl_int), &batch, sizeof(cl_int), &stride, sizeof(cl_int), &forward, sizeof(cl_int), &out, sizeof(cl_mem));
}


void mask_ongpu(int N, cl_mem  X, float mask_num, cl_mem  mask)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_mask_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &X, sizeof(cl_mem), &mask_num, sizeof(cl_int), &mask, sizeof(cl_mem));
}


void const_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_const_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void constrain_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_constrain_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void add_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_add_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void scal_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_scal_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void supp_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_supp_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void fill_ongpu(int N, float ALPHA, cl_mem  X, int INCX)
{
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(N);
	dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_fill_kernel, dimGrid, dimBlock, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, cl_mem out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);
    opencl_kernel(opencl_shortcut_kernel, dimGrid, dimBlock, 30, &size, sizeof(cl_int), &minw, sizeof(cl_int), &minh, sizeof(cl_int), &minc, sizeof(cl_int), &stride, sizeof(cl_int), &sample, sizeof(cl_int), &batch, sizeof(cl_int), &w1, sizeof(cl_int), &h1, sizeof(cl_int), &c1, sizeof(cl_int), &add, sizeof(cl_mem), &w2, sizeof(cl_int), &h2, sizeof(cl_int), &c2, sizeof(cl_int), &out, sizeof(cl_mem));
}


void smooth_l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);
    opencl_kernel(opencl_smooth_l1_kernel, dimN, dimBlock, 10, &n, sizeof(cl_int), &pred, sizeof(cl_mem), &truth, sizeof(cl_mem), &delta, sizeof(cl_mem), &error, sizeof(cl_mem));
}


void l2_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);
    opencl_kernel(opencl_l2_kernel, dimN, dimBlock, 10, &n, sizeof(cl_int), &pred, sizeof(cl_mem), &truth, sizeof(cl_mem), &delta, sizeof(cl_mem), &error, sizeof(cl_mem));
}


void l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);
    opencl_kernel(opencl_l1_kernel, dimN, dimBlock, 10, &n, sizeof(cl_int), &pred, sizeof(cl_mem), &truth, sizeof(cl_mem), &delta, sizeof(cl_mem), &error, sizeof(cl_mem));
}


void weighted_sum_gpu(cl_mem a, cl_mem b, cl_mem s, int num, cl_mem c)
{
    dim3 dimNum, dimBlock;
    dimNum = cuda_gridsize(num);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_weighted_sum_kernel, dimNum, dimBlock, 10, &num, sizeof(cl_int), &a, sizeof(cl_mem), &b, sizeof(cl_mem), &s, sizeof(cl_mem), &c, sizeof(cl_mem));
}


void weighted_delta_gpu(cl_mem a, cl_mem b, cl_mem s, cl_mem da, cl_mem db, cl_mem ds, int num, cl_mem dc)
{
    dim3 dimNum, dimBlock;
    dimNum = cuda_gridsize(num);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_weighted_delta_kernel, dimNum, dimBlock, 16, &num, sizeof(cl_int), &a, sizeof(cl_mem), &b, sizeof(cl_mem), &s, sizeof(cl_mem), &da, sizeof(cl_mem), &db, sizeof(cl_mem), &ds, sizeof(cl_mem), &dc, sizeof(cl_mem));
}


void mult_add_into_gpu(int num, cl_mem a, cl_mem b, cl_mem c)
{
    dim3 dimNum, dimBlock;
    dimNum = cuda_gridsize(num);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_mult_add_into_kernel, dimNum, dimBlock, 8, &num, sizeof(cl_int), &a, sizeof(cl_mem), &b, sizeof(cl_mem), &c, sizeof(cl_mem));
}

void softmax_offset_gpu(cl_mem input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output)
{
    dim3 dimBatch, dimBlock;
    dimBatch = cuda_gridsize(batch * groups);
    dimBlock = dim3_create(BLOCK, 1, 1);
    opencl_kernel(opencl_softmax_kernel, dimBatch, dimBlock, 20, &input, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &batch, sizeof(cl_int), &batch_offset, sizeof(cl_int), &groups, sizeof(cl_int), &group_offset, sizeof(cl_int), &stride, sizeof(cl_int), &temp, sizeof(cl_float), &output, sizeof(cl_mem));
}


void softmax_gpu(cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output)
{
    softmax_offset_gpu(input, 0, n, batch, batch_offset, groups, group_offset, stride, temp, output);
}

#endif // OPENCL