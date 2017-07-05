#include "catch.hpp"

//#include <stdio.h>
//
#include "blas.h"
//#include "cuda.h"
#include "unit.h"
#include "convolutional_layer.h"

TEST_CASE("Blas cpu gpu compare", "[blas][opencl]")
{
	const float threshold = 0.95;
	const size_t testSize = 416 * 416;
	size_t count = 0;

	// Allocate test arrays.
	float *A, *B, *C, *D;
	A = (float*) malloc(testSize * sizeof(float));
	B = (float*) malloc(testSize * sizeof(float));
	C = (float*) malloc(testSize * sizeof(float));
	D = (float*) malloc(testSize * sizeof(float));

	// Create random data.
	fillRandom(A, testSize);
	fillRandom(B, testSize);

	// Create opencl context.
	opencl_init(NULL, NULL, NULL);

	cl_mem A_gpu, B_gpu, C_gpu;

	A_gpu = cuda_make_array(A, testSize);
	B_gpu = cuda_make_array(B, testSize);
	C_gpu = cuda_make_array(C, testSize);
	
	SECTION("axpy 1")
	{
		axpy_cpu(testSize, 1.0, A, 1, C, 1);

		axpy_ongpu(testSize, 1.0, A_gpu, 1, C_gpu, 1);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);	
	}

	SECTION("axpy 2")
	{
		axpy_cpu(testSize, -1.0, A, 1, C, 1);

		axpy_ongpu(testSize, -1.0, A_gpu, 1, C_gpu, 1);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);
	}

	SECTION("axpy 3")
	{
		axpy_cpu(testSize, 0.1, A, 1, C, 1);

		axpy_ongpu(testSize, 0.1, A_gpu, 1, C_gpu, 1);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);
	}

	SECTION("copy")
	{
		copy_cpu(testSize, A, 1, C, 1);
		compare_array(A, C, testSize, threshold);

		copy_ongpu(testSize, A_gpu, 1, C_gpu, 1);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);
	}

	SECTION("scal 1")
	{
		scal_cpu(testSize, 1.0, A, 1);

		scal_ongpu(testSize, 1.0, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("scal 2")
	{
		scal_cpu(testSize, 2.0, A, 1);

		scal_ongpu(testSize, 2.0, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("scal 3")
	{
		scal_cpu(testSize, 0.1, A, 1);

		scal_ongpu(testSize, 0.1, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("fill 1")
	{
		fill_cpu(testSize, 1.0, A, 1);

		fill_ongpu(testSize, 1.0, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("fill 2")
	{
		fill_cpu(testSize, -1.0, A, 1);

		fill_ongpu(testSize, -1.0, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("fill 3")
	{
		fill_cpu(testSize, 0.1, A, 1);

		fill_ongpu(testSize, 0.1, A_gpu, 1);
		cuda_pull_array(A_gpu, C, testSize);

		compare_array(A, C, testSize, threshold);
	}

	SECTION("mean 1")
	{
		mean_cpu(A, 50, 50, 50, C);

		mean_gpu(A_gpu, 50, 50, 50, C_gpu);
		cuda_pull_array(C_gpu, D, 50 * 50 * 50);

		compare_array(C, D, 50 * 50 * 50, threshold);
	}

	SECTION("mean 2")
	{
		mean_cpu(A, 416, 416, 1, C);

		mean_gpu(A_gpu, 416, 416, 1, C_gpu);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);
	}

	SECTION("mean 3")
	{
		mean_cpu(A, 1, 416, 416, C);

		mean_gpu(A_gpu, 1, 416, 416, C_gpu);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);

	}

	SECTION("mean 4")
	{
		mean_cpu(A, 416, 1, 416, C);

		mean_gpu(A_gpu, 416, 1, 416, C_gpu);
		cuda_pull_array(C_gpu, D, testSize);

		compare_array(C, D, testSize, threshold);
	}

	SECTION("variance 1")
	{
		mean_cpu(A, 50, 50, 50, B);
		variance_cpu(A, B, 50, 50, 50, C);

		mean_gpu(A_gpu, 50, 50, 50, B_gpu);
		variance_gpu(A_gpu, B_gpu, 50, 50, 50, C_gpu);

		cuda_pull_array(C_gpu, D, 50 * 50 * 50);
		compare_array(C, D, 50 * 50 * 50, threshold);
	}

	SECTION("variance 2")
	{
		mean_cpu(A, 416, 416, 1, B);
		variance_cpu(A, B, 416, 416, 1, C);

		mean_gpu(A_gpu, 416, 416, 1, B_gpu);
		variance_gpu(A_gpu, B_gpu, 416, 416, 1, C_gpu);

		cuda_pull_array(C_gpu, D, testSize);
		compare_array(C, D, testSize, threshold);
	}

	SECTION("variance 3")
	{
		mean_cpu(A, 416, 1, 416, B);
		variance_cpu(A, B, 416, 1, 416, C);

		mean_gpu(A_gpu, 416, 1, 416, B_gpu);
		variance_gpu(A_gpu, B_gpu, 416, 1, 416, C_gpu);

		cuda_pull_array(C_gpu, D, testSize);
		compare_array(C, D, testSize, threshold);
	}

	SECTION("variance 4")
	{
		mean_cpu(A, 1, 416, 416, B);
		variance_cpu(A, B, 1, 416, 416, C);

		mean_gpu(A_gpu, 1, 416, 416, B_gpu);
		variance_gpu(A_gpu, B_gpu, 1, 416, 416, C_gpu);

		cuda_pull_array(C_gpu, D, testSize);
		compare_array(C, D, testSize, threshold);
	}

	SECTION("normalize 1")
	{
		mean_cpu(A, 1, 416, 416, B);
		variance_cpu(A, B, 1, 416, 416, C);
		normalize_cpu(A, B, C, 1, 416, 416);

		mean_gpu(A_gpu, 1, 416, 416, B_gpu);
		variance_gpu(A_gpu, B_gpu, 1, 416, 416, C_gpu);
		normalize_gpu(A_gpu, B_gpu, C_gpu, 1, 416, 416);

		cuda_pull_array(A_gpu, D, testSize);
		compare_array(A, D, testSize, threshold);
	}

	SECTION("normalize 2")
	{
		mean_cpu(A, 416, 1, 416, B);
		variance_cpu(A, B, 416, 1, 416, C);
		normalize_cpu(A, B, C, 416, 1, 416);

		mean_gpu(A_gpu, 416, 1, 416, B_gpu);
		variance_gpu(A_gpu, B_gpu, 416, 1, 416, C_gpu);
		normalize_gpu(A_gpu, B_gpu, C_gpu, 416, 1, 416);

		cuda_pull_array(A_gpu, D, testSize);
		compare_array(A, D, testSize, threshold);
	}

	SECTION("normalize 3")
	{
		mean_cpu(A, 416, 416, 1, B);
		variance_cpu(A, B, 416, 416, 1, C);
		normalize_cpu(A, B, C, 416, 416, 1);

		mean_gpu(A_gpu, 416, 416, 1, B_gpu);
		variance_gpu(A_gpu, B_gpu, 416, 416, 1, C_gpu);
		normalize_gpu(A_gpu, B_gpu, C_gpu, 416, 416, 1);

		cuda_pull_array(A_gpu, D, testSize);
		compare_array(A, D, testSize, threshold);
	}

	SECTION("normalize 4")
	{
		const int size = 8 * 32 * 173056;

		A = (float*) realloc(A, size * sizeof(float));
		B = (float*) realloc(B, size * sizeof(float));
		C = (float*) realloc(C, size * sizeof(float));
		D = (float*) realloc(D, size * sizeof(float));

		cuda_free(A_gpu);
		cuda_free(B_gpu);
		cuda_free(C_gpu);

		A_gpu = cuda_make_array(A, size);
		B_gpu = cuda_make_array(B, size);
		C_gpu = cuda_make_array(C, size);

		mean_cpu(A, 8, 32, 173056, B);
		variance_cpu(A, B, 8, 32, 173056, C);
		normalize_cpu(A, B, C, 8, 32, 173056);

		mean_gpu(A_gpu, 8, 32, 173056, B_gpu);
		variance_gpu(A_gpu, B_gpu, 8, 32, 173056, C_gpu);
		normalize_gpu(A_gpu, B_gpu, C_gpu, 8, 32, 173056);

		cuda_pull_array(A_gpu, D, size);
		compare_array(A, D, size, threshold);
	}

	SECTION("scale bias 1")
	{
		scale_bias(A, B, 416, 416, 1);
		scale_bias_gpu(A_gpu, B_gpu, 416, 416, 1);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("scale bias 2")
	{
		scale_bias(A, B, 416, 1, 416);
		scale_bias_gpu(A_gpu, B_gpu, 416, 1, 416);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("scale bias 3")
	{
		scale_bias(A, B, 1, 416, 416);
		scale_bias_gpu(A_gpu, B_gpu, 1, 416, 416);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("scale bias 4")
	{
		scale_bias(A, B, 104, 16, 104);
		scale_bias_gpu(A_gpu, B_gpu, 104, 16, 104);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("add bias 1")
	{
		add_bias(A, B, 416, 416, 1);
		add_bias_gpu(A_gpu, B_gpu, 416, 416, 1);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("add bias 2")
	{
		add_bias(A, B, 416, 1, 416);
		add_bias_gpu(A_gpu, B_gpu, 416, 1, 416);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("add bias 3")
	{
		add_bias(A, B, 1, 416, 416);
		add_bias_gpu(A_gpu, B_gpu, 1, 416, 416);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("add bias 4")
	{
		add_bias(A, B, 104, 16, 104);
		add_bias_gpu(A_gpu, B_gpu, 104, 16, 104);

		cuda_pull_array(A_gpu, C, testSize);
		compare_array(A, C, testSize, threshold);
	}

	SECTION("flatten 1")
	{
		flatten(A, 416, 416, 1, 1);
		flatten_ongpu(A_gpu, 416, 416, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 2")
	{
		flatten(A, 416, 416, 1, 0);
		flatten_ongpu(A_gpu, 416, 416, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 3")
	{
		flatten(A, 416, 1, 416, 1);
		flatten_ongpu(A_gpu, 416, 1, 416, 1, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 4")
	{
		flatten(A, 416, 1, 416, 0);
		flatten_ongpu(A_gpu, 416, 1, 416, 0, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 5")
	{
		flatten(A, 1, 416, 416, 1);
		flatten_ongpu(A_gpu, 1, 416, 416, 1, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 6")
	{
		flatten(A, 1, 416, 416, 0);
		flatten_ongpu(A_gpu, 1, 416, 416, 0, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 7")
	{
		flatten(A, 104, 16, 104, 1);
		flatten_ongpu(A_gpu, 104, 16, 104, 1, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("flatten 8")
	{
		flatten(A, 104, 16, 104, 0);
		flatten_ongpu(A_gpu, 104, 16, 104, 0, B_gpu);

		cuda_pull_array(B_gpu, B, testSize);
		compare_array(A, B, testSize, threshold);
	}

	SECTION("reorg 1")
	{
		reorg_cpu(A, 104, 104, 4, 4, 1, 0, B);
		reorg_ongpu(A_gpu, 104, 104, 4, 4, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 2")
	{
		reorg_cpu(A, 104, 104, 4, 4, 1, 1, B);
		reorg_ongpu(A_gpu, 104, 104, 4, 4, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 3")
	{
		reorg_cpu(A, 104, 4, 104, 4, 1, 0, B);
		reorg_ongpu(A_gpu, 104, 4, 104, 4, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 4")
	{
		reorg_cpu(A, 104, 4, 104, 4, 1, 1, B);
		reorg_ongpu(A_gpu, 104, 4, 104, 4, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 5")
	{
		reorg_cpu(A, 4, 104, 104, 4, 1, 0, B);
		reorg_ongpu(A_gpu, 4, 104, 104, 4, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 6")
	{
		reorg_cpu(A, 4, 104, 104, 4, 1, 1, B);
		reorg_ongpu(A_gpu, 4, 104, 104, 4, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 7")
	{
		reorg_cpu(A, 4, 4, 104, 104, 1, 0, B);
		reorg_ongpu(A_gpu, 4, 4, 104, 104, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 8")
	{
		reorg_cpu(A, 4, 4, 104, 104, 1, 1, B);
		reorg_ongpu(A_gpu, 4, 4, 104, 104, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 9")
	{
		reorg_cpu(A, 104, 4, 104, 4, 1, 0, B);
		reorg_ongpu(A_gpu, 104, 4, 104, 4, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 10")
	{
		reorg_cpu(A, 104, 4, 104, 4, 1, 1, B);
		reorg_ongpu(A_gpu, 104, 4, 104, 4, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 11")
	{
		reorg_cpu(A, 104, 4, 4, 104, 1, 0, B);
		reorg_ongpu(A_gpu, 104, 4, 4, 104, 1, 0, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 12")
	{
		reorg_cpu(A, 104, 4, 4, 104, 1, 1, B);
		reorg_ongpu(A_gpu, 104, 4, 4, 104, 1, 1, B_gpu);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("reorg 13")
	{
		const int size = 26 * 26 * 512 * 8;

		A = (float*) realloc(A, size * sizeof(float));
		B = (float*) realloc(B, size * sizeof(float));
		C = (float*) realloc(C, size * sizeof(float));

		fillRandom(A, size);

		cuda_free(A_gpu);
		cuda_free(B_gpu);

		A_gpu = cuda_make_array(A, size);
		B_gpu = cuda_make_array(B, size);

		reorg_cpu(A, 26, 26, 512, 8, 2, 0, B);
		reorg_ongpu(A_gpu, 26, 26, 512, 8, 2, 0, B_gpu);

		cuda_pull_array(B_gpu, C, size);
		compare_array(B, C, size, threshold);
	}

	SECTION("reorg 14")
	{
		const int size = 26 * 26 * 512 * 8;

		A = (float*) realloc(A, size * sizeof(float));
		B = (float*) realloc(B, size * sizeof(float));
		C = (float*) realloc(C, size * sizeof(float));

		fillRandom(A, size);

		cuda_free(A_gpu);
		cuda_free(B_gpu);

		A_gpu = cuda_make_array(A, size);
		B_gpu = cuda_make_array(B, size);

		reorg_cpu(A, 26, 26, 512, 8, 2, 1, B);
		reorg_ongpu(A_gpu, 26, 26, 512, 8, 2, 1, B_gpu);

		cuda_pull_array(B_gpu, C, size);
		compare_array(B, C, size, threshold);
	}

	SECTION("copy 1")
	{
		copy_cpu(testSize, A, 1, B, 1);
		copy_ongpu_offset(testSize, A_gpu, 0, 1, B_gpu, 0, 1);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B, C, testSize, threshold);
	}

	SECTION("copy 2")
	{
		const int offset = testSize / 2;

		copy_cpu(offset, A + offset, 1, B + offset, 1);
		copy_ongpu_offset(offset, A_gpu, offset, 1, B_gpu, offset, 1);

		cuda_pull_array(B_gpu, C, testSize);
		compare_array(B + offset, C + offset, offset, threshold);
	}



	cuda_free(A_gpu);
	cuda_free(B_gpu);
	cuda_free(C_gpu);
	opencl_deinit();
	free(A);
	free(B);
	free(C);
	free(D);
}