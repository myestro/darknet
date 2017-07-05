#ifndef CUDA_H
#define CUDA_H

extern int gpu_index;

#ifdef GPU

#ifdef OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#define GPU_DATA cl_mem
extern cl_command_queue opencl_queue;

typedef struct dim3_
{
    size_t x;
    size_t y;
    size_t z;
} dim3;

dim3 dim3_create(const size_t x, const size_t y, const size_t z);
#endif

#ifdef CUDA
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#define GPU_DATA float*
#endif 

#define BLOCK 512

#ifdef CUDNN
#include "cudnn.h"
#endif

#ifdef __cplusplus 
extern "C" {
#endif

// OpenCL 
#ifdef OPENCL
#define CONVERT_KERNEL_TO_STRING(...) #__VA_ARGS__
void opencl_load(const char *fileName, cl_program *output);
void opencl_load_buffer(const char *bufferName, const size_t size, cl_program *output);
void opencl_create_kernel(cl_program *program, const char *kernalName,
    cl_kernel *kernel);
void opencl_init(cl_context context, cl_command_queue queue, cl_device_id device);
void opencl_deinit();
void opencl_kernel(cl_kernel kernel, const dim3 globalItemSize,
    const dim3 localItemSize, const int argc, ...);
#endif

// CUDA
#ifdef CUDA
void check_error(cudaError_t status);
void check_cublas_error(cublasStatus_t status);
cublasHandle_t blas_handle();
#endif

GPU_DATA cuda_make_array(float *x, size_t n);

#if CUDA
int *cuda_make_int_array(size_t n);
#elif OPENCL
GPU_DATA cuda_make_int_array(size_t n);
#endif

void cuda_push_array(GPU_DATA x_gpu, float *x, size_t n);
void cuda_pull_array(GPU_DATA x_gpu, float *x, size_t n);
void cuda_set_device(int n);
void cuda_free(GPU_DATA x_gpu);
void cuda_random(GPU_DATA x_gpu, size_t n);
float cuda_compare(GPU_DATA x_gpu, float *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);
float cuda_mag_array(GPU_DATA x_gpu, size_t n);
void cuda_dump_mem_stat();

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
void cudnn_handle_reset();
void blas_handle_reset();
#endif

#ifdef __cplusplus 
}
#endif


#endif // GPU
#endif // CUDA_H
