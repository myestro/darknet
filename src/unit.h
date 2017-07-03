#ifndef DARKNET_UNIT_TEST_
#define DARKNET_UNIT_TEST_

#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "cuda.h"
#include "layer.h"
#include "network.h"

void fillRandom(float *array, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		array[i] = ((float) rand()) / ((float) (RAND_MAX));

		if (isnan(array[i])) array[i] = 1.0;
		if (isinf(array[i])) array[i] = 1.0;
	}
}

size_t compareArray(const float *a, const float *b, const size_t size)
{
	clFinish(opencl_queue);

	for (size_t i = 0; i < size; ++i)
	{
		//printf("fabs(%f - %f) = %f\n", a[i], b[i], fabs(a[i] - b[i]));
		REQUIRE(fabs(a[i] - b[i]) < FLT_EPSILON);
	}

	return size;
}

size_t compareArray2(const float *a, const float *b, const size_t size)
{
	clFinish(opencl_queue);
	size_t counter = 0;

	for (size_t i = 0; i < size; ++i)
	{
		if (fabs(a[i] - b[i]) < FLT_EPSILON)
		{
			counter++;
		}
	}

	return counter;
}

#define MYESTRO_EPSILON 0.02

void compare_array(const float *a, const float *b, const size_t size,
	const float threshold, const int print)
{
	clFinish(opencl_queue);

	float current = 0.0;
	float mean = 0.0;
	size_t counter = 0;

	for (size_t i = 0; i < size; ++i)
	{
		current = a[i] - b[i];

		if (fabs(current) < 10 * FLT_EPSILON)
		{
			counter++;
		}

		mean += current * current;

		if (print)
			printf("fabs(%f - %f) = %f\n", a[i], b[i], fabs(a[i] - b[i]));
	}

	float compare = ((float) counter) / ((float) size);

	WARN("Array comparision " << counter << "/" << size << "(" << compare * 100 << "%)\n"
		<< "Standard deviation: " << mean / ((float) size));
	CHECK(compare >= threshold);
}

#undef MYESTRO_EPSILON

layer* getLayer(network net, LAYER_TYPE type)
{
	for (int i = 0; i < net.n; ++i)
		if (net.layers[i].type == type)
			return &net.layers[i];

	return NULL;
}

#endif
