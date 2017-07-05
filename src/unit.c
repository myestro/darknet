#include <catch.hpp>
#include "unit.h"

#include <float.h>

void fillRandom(float *array, const size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = ((float) rand()) / ((float) (RAND_MAX));

        if (std::isnan(array[i])) array[i] = 1.0;
        if (std::isinf(array[i])) array[i] = 1.0;
    }
}

void compare_array(const float *a, const float *b,
    const size_t size, const float threshold)
{
    clFinish(opencl_queue);

    float current = 0.0;
    float deviation = 0.0;
    size_t counter = 0;

    for (size_t i = 0; i < size; ++i)
    {
        current = a[i] - b[i];

        if (fabs(current) < 10 * FLT_EPSILON)
        {
            counter++;
        }

     deviation += current * current; // (0 - current)^2
    }

    float compare = ((float) counter) / ((float) size);

    INFO("Array comparision " << counter << "/" << size << "(" << compare * 100 << "%)\n"
        << "Standard deviation: " << sqrt(deviation / ((float) size)));
    CHECK(compare >= threshold);
}

layer* getLayer(network net, LAYER_TYPE type)
{
    for (int i = 0; i < net.n; ++i)
        if (net.layers[i].type == type)
            return &net.layers[i];

    return NULL;
}