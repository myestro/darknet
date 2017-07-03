#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "gemm.h"
#include "cuda.h"
#include "unit.h"

float *randomMatrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows * cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float) rand() / RAND_MAX;
    }
    return m;
}


TEST_CASE("gemm kernel clash", "[gemm][opencl]")
{
	int TA = 0;
	int TB = 0;

	SECTION("gemm 1")
	{
		TA = 0;
		TB = 0;
	}

	SECTION("gemm 2")
	{
		TA = 0;
		TB = 1;
	}

	SECTION("gemm 3")
	{
		TA = 1;
		TB = 0;
	}

	SECTION("gemm 4")
	{
		TA = 1;
		TB = 1;
	}

	const int m = 1000;
	const int n = 100;
	const int k = 10;

	opencl_init(NULL, NULL, NULL);

	// Allocate test arrays.
	float *A, *B, *C, *D;
	A = (!TA) ? randomMatrix(m, k) : randomMatrix(k, m);
	B = (!TB) ? randomMatrix(k, n) : randomMatrix(n, k);
	C = randomMatrix(m, n);
	D = randomMatrix(m, n);

	int lda = (!TA) ? k : m;
	int ldb = (!TB) ? n : k;
	int ldc = 0;

	cl_mem A_gpu, B_gpu, C_gpu;

	A_gpu = cuda_make_array(A, m * k);
	B_gpu = cuda_make_array(B, k * n);
	C_gpu = cuda_make_array(C, m * n);

	gemm_cpu(TA, TB, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
	gemm_ongpu(TA, TB, m, n ,k, 1.0, A_gpu, 0, lda, B_gpu, 0, ldb, 1.0, C_gpu, 0, ldc);

	cuda_pull_array(C_gpu, D, m * n);
	size_t counter = compareArray2(C, D, m * n);

	CHECK(((float) counter) / ((float) m * n) > 0.9);
	printf("Comparison: %ld/%d (%5.2f%%)\n", counter, m * n, ((float) counter) / ((float) m * n) * 100);

	cuda_free(A_gpu);
	cuda_free(B_gpu);
	cuda_free(C_gpu);
	free(A);
	free(B);
	free(C);
	free(D);

	opencl_deinit();
}