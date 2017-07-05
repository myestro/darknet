#include "catch.hpp"

//#include <stdio.h>
//
#include "col2im.h"
#include "im2col.h"
//#include "cuda.h"
#include "unit.h"

TEST_CASE("col2im and im2col cpu gpu compare", "[col2im][opencl]")
{
	const size_t testSize = 416 * 416 * 3 * 3 * 3;
	const float threshold = 0.9;

	// Allocate test arrays.
	float *A, *B, *C;
	A = (float*) calloc(testSize, sizeof(float));
	B = (float*) calloc(testSize, sizeof(float));
	C = (float*) calloc(testSize, sizeof(float));

	// Create random data.
	fillRandom(A, testSize);

	// Create opencl context.
	opencl_init(NULL, NULL, NULL);

	cl_mem A_gpu, B_gpu, C_gpu;

	A_gpu = cuda_make_array(A, testSize);
	B_gpu = cuda_make_array(B, testSize);

	SECTION("col2im")
	{
		col2im_cpu(A, 3, 416, 416, 3, 1, 1, B);
		col2im_ongpu(A_gpu, 3, 416, 416, 3, 1, 1, B_gpu, 0);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("im2col")
	{
		im2col_cpu(A, 3, 416, 416, 3, 1, 1, B);
		im2col_ongpu(A_gpu, 0, 3, 416, 416, 3, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}


	cuda_free(A_gpu);
	cuda_free(B_gpu);
	opencl_deinit();
	free(A);
	free(B);
	free(C);
}