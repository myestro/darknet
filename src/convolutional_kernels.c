#include <string.h>

#include "activations.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "convolutional_kernels.cl"

#ifdef OPENCL

cl_kernel opencl_binarize_kernel;
cl_kernel opencl_binarize_input_kernel;
cl_kernel opencl_binarize_weights_kernel;
cl_kernel opencl_smooth_kernel;
cl_program opencl_convolutional_kernels_program;

void convolutional_kernel_init(void)
{
    opencl_load_buffer(convolutional_kernel_source, strlen(convolutional_kernel_source), &opencl_convolutional_kernels_program);

    opencl_create_kernel(&opencl_convolutional_kernels_program, "binarize_kernel", &opencl_binarize_kernel);
    opencl_create_kernel(&opencl_convolutional_kernels_program, "binarize_input_kernel", &opencl_binarize_input_kernel);
    opencl_create_kernel(&opencl_convolutional_kernels_program, "binarize_weights_kernel", &opencl_binarize_weights_kernel);
    opencl_create_kernel(&opencl_convolutional_kernels_program, "smooth_kernel", &opencl_smooth_kernel);

}

void convolutional_kernel_release(void)
{
    clReleaseKernel(opencl_binarize_kernel); opencl_binarize_kernel = 0;
    clReleaseKernel(opencl_binarize_input_kernel); opencl_binarize_input_kernel = 0;
    clReleaseKernel(opencl_binarize_weights_kernel); opencl_binarize_weights_kernel = 0;
    clReleaseKernel(opencl_smooth_kernel); opencl_smooth_kernel = 0;
    clReleaseProgram(opencl_convolutional_kernels_program); opencl_convolutional_kernels_program = 0;
}

void binarize_gpu(cl_mem x, int n, cl_mem binary)
{

    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_binarize_kernel, dimN, dimBlock, 6, &x, sizeof(cl_mem), &n, sizeof(cl_int), &binary, sizeof(cl_mem));
}


void binarize_input_gpu(cl_mem input, int n, int size, cl_mem binary)
{
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(size);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_binarize_input_kernel, dimN, dimBlock, 8, &input, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));    
}


void binarize_weights_gpu(cl_mem weights, int n, int size, cl_mem binary)
{
    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_binarize_weights_kernel, dimN, dimBlock, 8, &weights, sizeof(cl_mem), &n, sizeof(cl_int), &size, sizeof(cl_int), &binary, sizeof(cl_mem));    
}

void swap_binary_gpu(convolutional_layer *l)
{
    GPU_DATA swap_gpu = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap_gpu;
}

void forward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary_gpu(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
        swap_binary_gpu(&l);
        binarize_gpu(state.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        state.input_gpu = l.binary_input_gpu;
    }

    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        im2col_ongpu(state.input_gpu, i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace_gpu);
        cl_mem  a = l.weights_gpu;
        cl_mem  b = state.workspace_gpu;
        cl_mem  c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,0,k,b,0,n,1.,c,i*m*n,n);
    }

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary_gpu(&l);
}

//void forward_convolutional_layer2(convolutional_layer l, network_state state)
//{
//    int out_h = l.out_h;
//    int out_w = l.out_w;
//    int i;
//
//    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
//
//    if(l.xnor){
//        binarize_weights(l.weights_gpu, l.n, l.c*l.size*l.size, l.binary_weights_gpu);
//        swap_binary(&l);
//        binarize_cpu(state.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
//        state.input_gpu = l.binary_input_gpu;
//    }
//
//    int m = l.n;
//    int k = l.size*l.size*l.c;
//    int n = out_h*out_w;
//
//
//    cl_mem a = l.weights_gpu;
//    cl_mem b = state.workspace_gpu;
//    cl_mem c = l.output_gpu;
//
//    for(i = 0; i < l.batch; ++i){
//        im2col_cpu(state.input_gpu, l.c, l.h, l.w, 
//                l.size, l.stride, l.pad, b);
//        gemm_ongpu(0,0,m,n,k,1.,a,0,k,b,0,n,1.,c,0,n);
//    }
//
//    if(l.batch_normalize){
//        forward_batchnorm_layer(l, state);
//    } else {
//        add_bias(l.output_gpu, l.biases_gpu, l.batch, l.n, out_h*out_w);
//    }
//
//    activate_array(l.output_gpu, m*n*l.batch, l.activation);
//    if(l.binary || l.xnor) swap_binary(&l);
//}


void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    dim3 dimN, dimBlock;
    dimN = cuda_gridsize(n);
    dimBlock = dim3_create(BLOCK, 1, 1);

    opencl_kernel(opencl_smooth_kernel, dimN, dimBlock, 16, &l.output_gpu, sizeof(cl_mem), &n, sizeof(cl_int), &l.w, sizeof(cl_int), &l.h, sizeof(cl_int), &l.c, sizeof(cl_int), &size, sizeof(cl_int), &rate, sizeof(cl_float), &l.delta_gpu, sizeof(cl_mem));    
}


void backward_convolutional_layer_gpu(convolutional_layer l, network_state state)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    constrain_ongpu(l.outputs*l.batch, 1, state.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    cl_mem original_input = state.input_gpu;

    if(l.xnor) state.input_gpu = l.binary_input_gpu;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;
    for(i = 0; i < l.batch; ++i){
        cl_mem  a = l.delta_gpu;
        cl_mem  b = state.workspace_gpu;
        cl_mem  c = l.weight_updates_gpu;

        im2col_ongpu(state.input_gpu, i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace_gpu);
        gemm_ongpu(0,1,m,n,k,1,a, i*m*k,k,b,0,k,1,c,0,n);

        if(state.delta_gpu){
            if(l.binary || l.xnor) swap_binary_gpu(&l);
            cl_mem  a = l.weights_gpu;
            cl_mem  b = l.delta_gpu;
            cl_mem  c = state.workspace_gpu;

            gemm_ongpu(1,0,n,k,m,1,a,0,n,b, i*k*m,k,0,c,0,k);

            col2im_ongpu(state.workspace_gpu, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta_gpu, i*l.c*l.h*l.w);
            if(l.binary || l.xnor) {
                swap_binary_gpu(&l);
            }
            if(l.xnor) gradient_array_offset_ongpu(original_input, i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta_gpu);
        }
    }
}


void pull_convolutional_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}


void push_convolutional_layer(convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
    if (layer.adam){
        cuda_push_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_push_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}


void adam_update_gpu(cl_mem w, cl_mem d, cl_mem m, cl_mem v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_ongpu(n, B1, m, 1);
    scal_ongpu(n, B2, v, 1);
    axpy_ongpu(n, -decay*batch, w, 1, d, 1);

    axpy_ongpu(n, (1-B1), d, 1, m, 1);
    mul_ongpu(n, d, 1, d, 1);
    axpy_ongpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate/batch, eps, t);
    fill_ongpu(n, 0, d, 1);
}


void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
	int size = layer.size*layer.size*layer.c*layer.n;
	axpy_ongpu(layer.n, learning_rate / batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
	scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

	if (layer.scales_gpu){
		axpy_ongpu(layer.n, learning_rate / batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
		scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
	}

	if (layer.adam){
		scal_ongpu(size, layer.B1, layer.m_gpu, 1);
		scal_ongpu(size, layer.B2, layer.v_gpu, 1);

		axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

		axpy_ongpu(size, -(1 - layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
		mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
		axpy_ongpu(size, (1 - layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

		adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate / batch, layer.eps, layer.t + 1);
		fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
	}
	else{
		axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
		axpy_ongpu(size, learning_rate / batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
		scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
	}
}

#endif // OpenCL