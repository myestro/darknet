#include <string.h>
#include "activations.h"
#include "cuda.h"
#include "activation_kernels.cl"

#ifdef OPENCL

cl_program opencl_activation_kernel_program = 0;
cl_kernel opencl_activate_array_kernel = 0;
cl_kernel opencl_gradient_array_kernel = 0;

void activation_kernels_init(void)
{
	opencl_load_buffer(activation_kernels_source, strlen(activation_kernels_source), &opencl_activation_kernel_program);
	opencl_create_kernel(&opencl_activation_kernel_program,
		"activate_array_kernel", &opencl_activate_array_kernel);
	opencl_create_kernel(&opencl_activation_kernel_program,
		"gradient_array_kernel", &opencl_gradient_array_kernel);
}

void activation_kernels_release(void)
{
	clReleaseKernel(opencl_activate_array_kernel);
	clReleaseKernel(opencl_gradient_array_kernel);
	clReleaseProgram(opencl_activation_kernel_program);

	opencl_activate_array_kernel = 0;
	opencl_gradient_array_kernel = 0;
	opencl_activation_kernel_program = 0;
}

void activate_array_offset_ongpu(cl_mem x, int offset, int n, ACTIVATION a) 
{
	dim3 dimN, dimBlock;
	dimN = cuda_gridsize(n);
	dimBlock = dim3_create(BLOCK, 1, 1);
	opencl_kernel(opencl_activate_array_kernel, dimN, dimBlock, 8, &x, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int));
}

void activate_array_ongpu(cl_mem x, int n, ACTIVATION a) 
{
    activate_array_offset_ongpu(x, 0, n, a);
}

void gradient_array_offset_ongpu(cl_mem x, int offset, int n, ACTIVATION a, cl_mem delta)
{
	dim3 dimN, dimBlock;
	dimN = cuda_gridsize(n);
	dimBlock = dim3_create(BLOCK, 1, 1);
	opencl_kernel(opencl_gradient_array_kernel, dimN, dimBlock, 10, &x, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int), &delta, sizeof(cl_mem));
}

void gradient_array_ongpu(cl_mem x, int n, ACTIVATION a, cl_mem delta) 
{
    gradient_array_offset_ongpu(x, 0, n, a, delta);
}

#endif // OPENCL