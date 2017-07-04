#ifndef __SOFTMAX_LAYER_KERNELS_CL__
#define __SOFTMAX_LAYER_KERNELS_CL__

static const char* const softmax_layer_kernel_source = CONVERT_KERNEL_TO_STRING(

__kernel void forward_softmax_layer_kernel(int n, int batch, __global float *input, float temp, __global float *output)
{
    int b = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(b >= batch) return;

    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i+b*n];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        sum += exp(input[i+b*n]/temp-largest/temp);
    }
    sum = (sum != 0) ? largest/temp+log(sum) : largest-100;
    for(i = 0; i < n; ++i){
        output[i+b*n] = exp(input[i+b*n]/temp-sum);
    }
}
);

#endif