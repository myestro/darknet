#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "convolutional_layer.h"
#include "cuda.h"
#include "unit.h"
#include "parser.h"
#include "im2col.h"
#include "blas.h"
#include "gemm.h"
#include "region_layer.h"

extern int entry_index(layer l, int batch, int location, int entry);

TEST_CASE("convolutional kernal bash", "[convolutional][opencl]")
{
	const float threshold = 0.9;

	opencl_init(NULL, NULL, NULL);

	network net = load_network("yolo.cfg", "yolo.weights", 0);

	network_state state;
	state.net = net;
	state.index = 0;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	state.workspace = net.workspace;
	state.workspace_gpu = net.workspace_gpu;

	layer *l = getLayer(net, REGION);

	REQUIRE(l != NULL);

	const size_t testSize = l->batch * l->outputs;

	// Allocate test arrays.
	float *A, *B;
	A = (float*) malloc(testSize * sizeof(float));
	B = (float*) malloc(testSize * sizeof(float));

	// Create random data.
	fillRandom(A, testSize);

	cl_mem A_gpu;

	A_gpu = cuda_make_array(A, testSize);

	state.input = A;
	state.input_gpu = A_gpu;

	SECTION("forward region composite")
	{
		memcpy(l->output, state.input, l->outputs*l->batch*sizeof(float));
		copy_ongpu(l->batch*l->inputs, state.input_gpu, 1, l->output_gpu, 1);

		cuda_pull_array(l->output_gpu, B, l->batch*l->inputs);
		compare_array(l->output, B, l->batch * l->inputs, threshold, 0);

	    int i,j,b,t,n;

	    for (b = 0; b < l->batch; ++b){
	        for(n = 0; n < l->n; ++n){
	            int index = entry_index(*l, b, n*l->w*l->h, 0);

	            activate_array(l->output + index, 2*l->w*l->h, LOGISTIC);
	            activate_array_offset_ongpu(l->output_gpu, index, 2*l->w*l->h, LOGISTIC);

	            index = entry_index(*l, b, n*l->w*l->h, 4);

	            activate_array(l->output + index,   l->w*l->h, LOGISTIC);
	            activate_array_offset_ongpu(l->output_gpu, index,   l->w*l->h, LOGISTIC);
	        }
	    }

	    cuda_pull_array(l->output_gpu, B, l->batch * l->n * l->w * l->h);
	    compare_array(l->output, B, l->batch * l->n * l->w * l->h, threshold, 0);

	    if (l->softmax_tree){
	        int i;
	        int count = 5;
	        for (i = 0; i < l->softmax_tree->groups; ++i) {
	            int group_size = l->softmax_tree->group_size[i];
	            int index = entry_index(*l, 0, 0, count);

	            softmax_cpu(state.input + count, group_size, l->batch, l->inputs, l->n*l->w*l->h, 1, l->n*l->w*l->h, l->temperature, l->output + count);
	            softmax_offset_gpu(state.input_gpu, index, group_size, l->batch*l->n, l->inputs/l->n, l->w*l->h, 1, l->w*l->h, 1, l->output_gpu);
	            count += group_size;
	        }

	        cuda_pull_array(l->output_gpu, B, l->batch * l->inputs);
	        compare_array(l->output, B, l->batch * l->inputs, threshold, 0);

	    } else if (l->softmax) {
	        int index = entry_index(*l, 0, 0, 5);

	        softmax_cpu(state.input + index, l->classes, l->batch*l->n, l->inputs/l->n, l->w*l->h, 1, l->w*l->h, 1, l->output + index);
	        softmax_offset_gpu(state.input_gpu, index, l->classes, l->batch*l->n, l->inputs/l->n, l->w*l->h, 1, l->w*l->h, 1, l->output_gpu);

	        cuda_pull_array(l->output_gpu, B, l->batch * l->outputs);
	        compare_array(l->output, B, l->batch * l->outputs, threshold, 0);
	    }
	    if(!state.train || l->onlyforward){
	        //cuda_pull_array(l->output_gpu, l->output, l->batch*l->outputs);
	    }	

	    memset(l->delta, 0, l->outputs * l->batch * sizeof(float));

	    cuda_pull_array(l->output_gpu, B, l->batch * l->outputs);
	    compare_array(l->output, B, l->batch * l->outputs, threshold, 0);
	}

	SECTION("Forward region versus")
	{
		gpu_index = -1;
		forward_region_layer(*l, state);

		memcpy(B, l->output, l->batch * l->outputs * sizeof(float));

		gpu_index = 1;
		forward_region_layer_gpu(*l, state);

		cuda_pull_array(l->output_gpu, A, l->batch * l->outputs);
		compare_array(A, B, l->batch * l->outputs, threshold, 0);
	}

	free_network(net);
	opencl_deinit();
	free(A);
	free(B);
}