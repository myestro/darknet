#ifdef OPENCL

#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"

#include "dropout_layer_kernels.cl"

cl_program opencl_dropout_layer_program = 0;
cl_kernel opencl_yoloswag420blazeit360noscopemMmMmMonsterKill = 0;

void dropout_kernel_init(void)
{
    opencl_load_buffer(dropout_layer_kernel_source, strlen(dropout_layer_kernel_source), &opencl_dropout_layer_program);

    opencl_create_kernel(&opencl_dropout_layer_program, "yoloswag420blazeit360noscopemMmMmMonsterKill", &opencl_yoloswag420blazeit360noscopemMmMmMonsterKill);
}

void dropout_kernel_release(void)
{
    clReleaseKernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill);
    opencl_yoloswag420blazeit360noscopemMmMmMonsterKill = 0;

    clReleaseProgram(opencl_dropout_layer_program);
}

void forward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if (!state.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);

    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill, dimGrid, dimBlock, 10, &state.input, sizeof(cl_mem), &size, sizeof(cl_int), &layer.rand_gpu, sizeof(cl_mem), &layer.probability, sizeof(cl_float), &layer.scale, sizeof(cl_float));
}

void backward_dropout_layer_gpu(dropout_layer layer, network_state state)
{
    if(!state.delta) return;
    int size = layer.inputs*layer.batch;

    dim3 dimGrid, dimBlock;
    dimGrid = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill, dimGrid, dimBlock, 10, &state.delta_gpu, sizeof(cl_mem), &size, sizeof(cl_int), &layer.rand_gpu, sizeof(cl_mem), &layer.probability, sizeof(cl_float), &layer.scale, sizeof(cl_float));
}

#endif