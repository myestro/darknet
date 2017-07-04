#ifndef BLAS_H
#define BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

//void reorg(float *x, int size, int layers, int batch, int forward);

void flatten(float *x, int size, int layers, int batch, int forward);

void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
#ifdef __cplusplus
}
#endif


#ifdef GPU
#include "cuda.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCL
void blas_kernel_init(void);
void blas_kernel_release(void);
#endif

void axpy_ongpu(int N, float ALPHA, GPU_DATA X, int INCX, GPU_DATA Y, int INCY);
void axpy_ongpu_offset(int N, float ALPHA, GPU_DATA X, int OFFX, int INCX, GPU_DATA Y, int OFFY, int INCY);
void copy_ongpu(int N, GPU_DATA X, int INCX, GPU_DATA Y, int INCY);
void copy_ongpu_offset(int N, GPU_DATA X, int OFFX, int INCX, GPU_DATA Y, int OFFY, int INCY);
void scal_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);
void add_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);
void supp_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);
void mask_ongpu(int N, GPU_DATA X, float mask_num, GPU_DATA mask);
void const_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);
void const_ongpu_offset(int N, float ALPHA, GPU_DATA X, int OFFX, int INCX);
void pow_ongpu(int N, float ALPHA, GPU_DATA X, int INCX, GPU_DATA Y, int INCY);
void pow_ongpu_offset(int N, float ALPHA, GPU_DATA X, int OFFX, int INCX, GPU_DATA Y, int OFFY, int INCY);
void mul_ongpu(int N, GPU_DATA X, int INCX, GPU_DATA Y, int INCY);
void fill_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);
void constrain_ongpu(int N, float ALPHA, GPU_DATA X, int INCX);

void mean_gpu(GPU_DATA x, int batch, int filters, int spatial, GPU_DATA mean);
void variance_gpu(GPU_DATA x, GPU_DATA mean, int batch, int filters, int spatial, GPU_DATA variance);
void normalize_gpu(GPU_DATA x, GPU_DATA mean, GPU_DATA variance, int batch, int filters, int spatial);

void normalize_delta_gpu(GPU_DATA x, GPU_DATA mean, GPU_DATA variance, GPU_DATA mean_delta, GPU_DATA variance_delta, int batch, int filters, int spatial, GPU_DATA delta);

void fast_mean_delta_gpu(GPU_DATA delta, GPU_DATA variance, int batch, int filters, int spatial, GPU_DATA mean_delta);
void fast_variance_delta_gpu(GPU_DATA x, GPU_DATA delta, GPU_DATA mean, GPU_DATA variance, int batch, int filters, int spatial, GPU_DATA variance_delta);

void fast_variance_gpu(GPU_DATA x, GPU_DATA mean, int batch, int filters, int spatial, GPU_DATA variance);
void fast_mean_gpu(GPU_DATA x, int batch, int filters, int spatial, GPU_DATA mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, GPU_DATA add, int w2, int h2, int c2, GPU_DATA out);
void scale_bias_gpu(GPU_DATA output, GPU_DATA biases, int batch, int n, int size);
void scale_bias_gpu2(GPU_DATA output, GPU_DATA biases, int batch, int n, int size);
void backward_scale_gpu(GPU_DATA x_norm, GPU_DATA delta, int batch, int n, int size, GPU_DATA scale_updates);
void add_bias_gpu(GPU_DATA output, GPU_DATA biases, int batch, int n, int size);
void add_bias_gpu2(GPU_DATA output, GPU_DATA biases, int batch, int n, int size);
void backward_bias_gpu(GPU_DATA bias_updates, GPU_DATA delta, int batch, int n, int size);

void smooth_l1_gpu(int n, GPU_DATA pred, GPU_DATA truth, GPU_DATA delta, GPU_DATA error);
void l2_gpu(int n, GPU_DATA pred, GPU_DATA truth, GPU_DATA delta, GPU_DATA error);
void l1_gpu(int n, GPU_DATA pred, GPU_DATA truth, GPU_DATA delta, GPU_DATA error);
void weighted_delta_gpu(GPU_DATA a, GPU_DATA b, GPU_DATA s, GPU_DATA da, GPU_DATA db, GPU_DATA ds, int num, GPU_DATA dc);
void weighted_sum_gpu(GPU_DATA a, GPU_DATA b, GPU_DATA s, int num, GPU_DATA c);
void mult_add_into_gpu(int num, GPU_DATA a, GPU_DATA b, GPU_DATA c);

void reorg_ongpu(GPU_DATA x, int w, int h, int c, int batch, int stride, int forward, GPU_DATA out);

void softmax_gpu(GPU_DATA input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, GPU_DATA output);
void softmax_offset_gpu(GPU_DATA input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, GPU_DATA output);
void adam_gpu(int n, GPU_DATA x, GPU_DATA m, GPU_DATA v, float B1, float B2, float rate, float eps, int t);
void flatten_ongpu(GPU_DATA x, int spatial, int layers, int batch, int forward, GPU_DATA out);

#ifdef __cplusplus
}
#endif


#endif



#endif
