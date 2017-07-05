#include "catch.hpp"

//#include <stdio.h>
//#include <math.h>
//#include <float.h>
//
#include "convolutional_layer.h"
//#include "cuda.h"
#include "unit.h"
#include "parser.h"
#include "im2col.h"
#include "blas.h"
#include "gemm.h"

extern void binarize_gpu(cl_mem x, int n, cl_mem binary);
extern void binarize_cpu(float *x, int n, float *binary);
extern void binarize_input(float *x, int n, int size, float *binary);
extern void binarize_input_gpu(cl_mem x, int n, int size, cl_mem binary);
extern void binarize_weights(float *x, int n, int size, float *binary);
extern void binarize_weights_gpu(cl_mem x, int n, int size, cl_mem binary);

TEST_CASE("Convolutional layer cpu gpu compare", "[convolutional][opencl]")
{
	const float threshold = 0.9;

	opencl_init(NULL, NULL, NULL);

	network net = load_network(yolo_configuration_file, yolo_weights_file, 0);

	network_state state;
	state.net = net;
	state.index = 0;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	state.workspace = net.workspace;
	state.workspace_gpu = net.workspace_gpu;

	layer *l = getLayer(net, CONVOLUTIONAL);

	REQUIRE(l != NULL);

	const size_t testSize = l->batch * l->outputs;

	// Allocate test arrays.
	float *A, *B, *C;
	A = (float*) malloc(testSize * sizeof(float));
	B = (float*) malloc(testSize * sizeof(float));
	C = (float*) malloc(testSize * sizeof(float));

	// Create random data.
	fillRandom(A, testSize);

	cl_mem A_gpu, B_gpu, original;

	A_gpu = cuda_make_array(A, testSize);
	B_gpu = cuda_make_array(B, testSize);

	state.input = A;
	state.input_gpu = A_gpu;

	SECTION("binarize")
	{
		binarize_cpu(A, testSize, B);
		binarize_gpu(A_gpu, testSize, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("binarize 2")
	{
		binarize_input(A, l->n, l->c * l->size * l->size, B);
		binarize_input_gpu(A_gpu, l->n, l->c * l->size * l->size, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("binarize 3")
	{
		binarize_weights(A, l->n, l->c * l->size * l->size, B);
		binarize_weights_gpu(A_gpu, l->n, l->c * l->size * l->size, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

//  Not sure how smooth works atm.
//	SECTION("smooth")
//	{
//		
//	}

	SECTION("forward convolutional")
	{
		forward_convolutional_layer(*l, state);
		forward_convolutional_layer_gpu(*l, state);

		cuda_pull_array(l->output_gpu, A, l->outputs * l->batch);
		compare_array(l->output, A, l->outputs * l->batch, threshold);
	}

	// Composite comparission of the cpu and gpu path.
	// With binary = 0
	// xnor = 0
	// batch_normalize = 1
	// And with batch_normalize unrolled
	// l->type = CONVOLUTIONAL
	// state.train = 0
	SECTION("forward convolutional extended.")
	{
		REQUIRE(l->binary == 0);
		REQUIRE(l->xnor == 0);
		REQUIRE(l->batch_normalize == 1);
		REQUIRE(state.train == 0);

		size_t count = 0;

		fill_cpu(l->outputs*l->batch, 0, l->output, 1);
		fill_ongpu(l->outputs*l->batch, 0, l->output_gpu, 1);

		cuda_pull_array(l->output_gpu, A, l->outputs*l->batch);
		compare_array(l->output, A, l->outputs*l->batch, threshold);

		// These values are the same for CPU and GPU
		int i;
    	int m = l->n;
    	int k = l->size * l->size * l->c;
    	int n = l->out_w * l->out_h;		

    	// CPU specific
    	float *a_cpu = l->weights;
    	float *b_cpu = state.workspace;
    	float *c_cpu = l->output;

    	// compare inputs.
    	cuda_pull_array(state.input_gpu, A, l->batch * l->outputs);
    	compare_array(state.input, A, l->batch * l->outputs, threshold);

    	for (i = 0; i < l->batch; ++i)
    	{
    		im2col_cpu(state.input, l->c, l->h, l->w, l->size, l->stride, l->pad, b_cpu);
    		im2col_ongpu(state.input_gpu, i*l->c*l->h*l->w, l->c,  l->h,  l->w,  l->size,  l->stride, l->pad, state.workspace_gpu);
			
			//cuda_pull_array(state.workspace_gpu, A, l->c*l->h*l->w);
			//compare_array(b_cpu, A, l->c*l->h*l->w, threshold);

			// GPU specific
			cl_mem a_gpu = l->weights_gpu;
        	cl_mem b_gpu = state.workspace_gpu;
        	cl_mem c_gpu = l->output_gpu;

        	gemm(0,0,m,n,k,1,a_cpu,k,b_cpu,n,1,c_cpu,n);
        	gemm_ongpu(0,0,m,n,k,1.,a_gpu,0,k,b_gpu,0,n,1.,c_gpu,i*m*n,n);

        	//cuda_pull_array(c_gpu, A, m*n);
        	//compare_array(c_cpu, A, m*n, threshold);

        	// CPU specific
        	c_cpu += n*m;
        	state.input += l->c*l->h*l->w;	
    	}

    	cuda_pull_array(l->output_gpu, A, m*n*l->batch);
        compare_array(l->output, A, m*n*l->batch, threshold);

    	// Unroll of the batchnorm layer.
		cuda_pull_array(l->rolling_mean_gpu, A, l->out_c);
        compare_array(l->rolling_mean, A, l->out_c, threshold);

        cuda_pull_array(l->rolling_variance_gpu, A, l->out_c);
        compare_array(l->rolling_variance, A, l->out_c, threshold);

    	normalize_cpu(l->output, l->rolling_mean, l->rolling_variance, l->batch, l->out_c, l->out_h*l->out_w);
    	normalize_gpu(l->output_gpu, l->rolling_mean_gpu, l->rolling_variance_gpu, l->batch, l->out_c, l->out_h*l->out_w);

    	cuda_pull_array(l->output_gpu, A, l->batch * l->c * l->out_w * l->out_h);
    	compare_array(l->output, A, l->batch * l->c * l->out_w * l->out_h, threshold);

    	cuda_pull_array(l->scales_gpu, A, l->out_c);
        compare_array(l->scales, A, l->out_c, threshold);

        scale_bias(l->output, l->scales, l->batch, l->out_c, l->out_h*l->out_w);
        scale_bias_gpu(l->output_gpu, l->scales_gpu, l->batch, l->out_c, l->out_h*l->out_w);

        cuda_pull_array(l->output_gpu, A, l->batch * l->c * l->out_w * l->out_h);
    	compare_array(l->output, A, l->batch * l->c * l->out_w * l->out_h, threshold);

    	cuda_pull_array(l->biases_gpu, A, l->out_c);
        compare_array(l->biases, A, l->out_c, threshold);

        add_bias(l->output, l->biases, l->batch, l->out_c, l->out_h*l->out_w);
        add_bias_gpu(l->output_gpu, l->biases_gpu, l->batch, l->out_c, l->out_w*l->out_h);

        cuda_pull_array(l->output_gpu, A, l->batch * l->c * l->out_w * l->out_h);
    	compare_array(l->output, A, l->batch * l->c * l->out_w * l->out_h, threshold);
        // End unroll of the batchnorm layer.

		activate_array(l->output, m*n*l->batch, l->activation);
		activate_array_ongpu(l->output_gpu, l->outputs*l->batch, l->activation);

		cuda_pull_array(l->output_gpu, A, l->outputs*l->batch);
		compare_array(l->output, A, l->outputs*l->batch, threshold);
	}



    //cuda_free(state.input_gpu);

    free(A);
    free(B);
    free(C);
    free_network(net);
	opencl_deinit();
}