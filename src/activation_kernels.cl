#ifndef __ACTIVATION_KERNELS_CL__
#define __ACTIVATION_KERNELS_CL__

static const char* const activation_kernels_source = CONVERT_KERNEL_TO_STRING(
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1./(1. + exp(-x));}
float loggy_activate_kernel(float x){return 2./(1. + exp(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1*x;}
float leaky_activate_kernel(float x){return (x>0) ? x : .1*x;}
float tanh_activate_kernel(float x){return (2/(1 + exp(-2*x)) - 1);}
float plse_activate_kernel(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}
float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
 

float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}
float loggy_gradient_kernel(float x)
{
   float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01;}
float ramp_gradient_kernel(float x){return (x>0)+.1;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01 : .125;}
float stair_gradient_kernel(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}

float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

float gradient_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__kernel void activate_array_kernel(__global float *x, int offset, int n, ACTIVATION a)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n) x[i + offset] = activate_kernel(x[i + offset], a);
}

__kernel void gradient_array_kernel(__global float *x, int offset, int n, ACTIVATION a, __global float *delta)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n) delta[i + offset] *= gradient_kernel(x[i + offset], a);
}
);
#endif