#include "softmax_layer.h"
#include "cuda.h"
#include "blas.h"

#include "softmax_layer_kernels.cl"

cl_program opencl_softmax_layer_kernel_program = 0;
cl_kernel opencl_forward_softmax_layer_kernel = 0;

void softmax_kernel_init(void)
{
  opencl_load_buffer(softmax_layer_kernel_source, strlen(softmax_layer_kernel_source), &opencl_softmax_layer_kernel_program);

  opencl_create_kernel(&opencl_softmax_layer_kernel_program, "forward_softmax_layer_kernel", &opencl_forward_softmax_layer_kernel);
}

void softmax_kernel_release(void)
{
  clReleaseKernel(opencl_forward_softmax_layer_kernel);
  opencl_forward_softmax_layer_kernel = 0;

  clReleaseProgram(opencl_softmax_layer_kernel_program);
  opencl_softmax_layer_kernel_program = 0;
}

void forward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    int inputs = layer.inputs / layer.groups;
    int batch = layer.batch * layer.groups;

    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(batch);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_forward_softmax_layer_kernel, dimGrid, dimBlock, 10, &inputs, sizeof(cl_int), &batch, sizeof(cl_int), &state.input_gpu, sizeof(cl_mem), &layer.temperature, sizeof(cl_float), &layer.output_gpu, sizeof(cl_mem));
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}