#include "catch.hpp"

//#include <stdio.h>
//
#include "maxpool_layer.h"
//#include "cuda.h"
#include "unit.h"

TEST_CASE("Maxpool cpu gpu compare", "[maxpool][opencl]")
{
	opencl_init(NULL, NULL, NULL);

	network net = load_network(yolo_configuration_file, yolo_weights_file, 0);

	network_state state;
	state.net = net;
	state.index = 0;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;

	layer *l = getLayer(net, MAXPOOL);

	REQUIRE(l != NULL);

	const size_t testSize = l->h * l->w * l->c * l->batch;
	const float threshold = 0.9;

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

		cuda_pull_array(l->output_gpu, C, l->outputs*l->batch);

		compare_array(l->output, C, testSize, threshold);
	}

	cuda_free(A_gpu);
	free_network(net);
	opencl_deinit();
	free(A);
	free(B);
	free(C);
}