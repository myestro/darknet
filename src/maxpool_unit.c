#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdio.h>

#include "maxpool_layer.h"
#include "cuda.h"
#include "unit.h"

TEST_CASE("max pool kernel bash", "[maxpool][opencl]")
{
	opencl_init(NULL, NULL, NULL);

	network net = load_network("yolo.cfg", "yolo.weights", 0);

	network_state state;
	state.net = net;
	state.index = 0;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;

	layer *l = getLayer(net, MAXPOOL);

	REQUIRE(l != NULL);

	const size_t testSize = l->h * l->w * l->c * l->batch;

	// Allocate test arrays.
	float *A, *B, *C;
	A = (float*) malloc(testSize * sizeof(float));
	B = (float*) malloc(testSize * sizeof(float));
	C = (float*) malloc(testSize * sizeof(float));

	// Create random data.
	fillRandom(A, testSize);

	cl_mem A_gpu;

	A_gpu = cuda_make_array(A, testSize);

	state.input = A;
	state.input_gpu = A_gpu;

	SECTION("Forward")
	{
		forward_maxpool_layer(*l, state);
		forward_maxpool_layer_gpu(*l, state);

		clFinish(opencl_queue);

		B = l->output;
		cuda_pull_array(l->output_gpu, C, l->outputs*l->batch);

		compareArray(B, C, testSize);
	}

//  Backward CPU code is not implemented oO.
//	SECTION("Backward")
//	{
//		backward_maxpool_layer(*l, state);
//		backward_maxpool_layer_gpu(*l, state);
//
//		clFinish(opencl_queue);
//
//		B = l->output;
//		cuda_pull_array(l->output_gpu, C, l->outputs*l->batch);
//
//		compareArray(B, C, testSize);
//	}

	opencl_deinit();

	cuda_free(A_gpu);
	free(A);
	free(B);
	free(C);
	free_network(net);
}