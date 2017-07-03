#include <string.h>

#include "maxpool_layer.h"
#include "cuda.h"
#include "maxpool_layer_kernels.cl"

#ifdef OPENCL

cl_kernel opencl_forward_maxpool_layer_kernel;
cl_kernel opencl_backward_maxpool_layer_kernel;
cl_program opencl_maxpool_layer_kernels_program;

void maxpool_kernel_init(void)
{
	opencl_load_buffer(maxpool_kernel_source, strlen(maxpool_kernel_source), &opencl_maxpool_layer_kernels_program);
	opencl_create_kernel(&opencl_maxpool_layer_kernels_program, "forward_maxpool_layer_kernel", &opencl_forward_maxpool_layer_kernel);
	opencl_create_kernel(&opencl_maxpool_layer_kernels_program, "backward_maxpool_layer_kernel", &opencl_backward_maxpool_layer_kernel);
}

void maxpool_kernel_release(void)
{
	clReleaseKernel(opencl_forward_maxpool_layer_kernel); opencl_forward_maxpool_layer_kernel = 0;
	clReleaseKernel(opencl_backward_maxpool_layer_kernel); opencl_backward_maxpool_layer_kernel = 0;	
	clReleaseProgram(opencl_maxpool_layer_kernels_program); opencl_maxpool_layer_kernels_program = 0;
}

void forward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_forward_maxpool_layer_kernel, dimN, dimBlock, 20, &n, sizeof(cl_int), &layer.h, sizeof(cl_int), &layer.w, sizeof(cl_int), &layer.c, sizeof(cl_int), &layer.stride, sizeof(cl_int), &layer.size, sizeof(cl_int), &layer.pad, sizeof(cl_int), &state.input_gpu, sizeof(cl_mem), &layer.output_gpu, sizeof(cl_mem), &layer.indexes_gpu, sizeof(cl_mem));
}

void backward_maxpool_layer_gpu(maxpool_layer layer, network_state state)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_backward_maxpool_layer_kernel, dimN, dimBlock, 20, &n, sizeof(cl_int), &layer.h, sizeof(cl_int), &layer.w, sizeof(cl_int), &layer.c, sizeof(cl_int), &layer.stride, sizeof(cl_int), &layer.size, sizeof(cl_int), &layer.pad, sizeof(cl_int), &layer.delta_gpu, sizeof(cl_mem), &state.delta_gpu, sizeof(cl_mem), &layer.indexes_gpu, sizeof(cl_mem));
}

#endif // OPENCL