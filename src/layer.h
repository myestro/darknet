#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"
#include "stddef.h"
#include "tree.h"

struct network_state;

struct layer;
typedef struct layer layer;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    REORG,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SMOOTH
} COST_TYPE;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network_state);
    void (*backward)  (struct layer, struct network_state);
    void (*update)    (struct layer, int, float, float, float);
    void (*forward_gpu)   (struct layer, struct network_state);
    void (*backward_gpu)  (struct layer, struct network_state);
    void (*update_gpu)    (struct layer, int, float, float, float);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;

    int adam;
    float B1;
    float B2;
    float eps;
    int t;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;
    int bias_match;
    int random;
    float thresh;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * z_cpu;
    float * r_cpu;
    float * h_cpu;

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    tree *softmax_tree;

    size_t workspace_size;

    #ifdef GPU
    GPU_DATA indexes_gpu;

    GPU_DATA z_gpu;
    GPU_DATA r_gpu;
    GPU_DATA h_gpu;

    GPU_DATA m_gpu;
    GPU_DATA v_gpu;

    GPU_DATA prev_state_gpu;
    GPU_DATA forgot_state_gpu;
    GPU_DATA forgot_delta_gpu;
    GPU_DATA state_gpu;
    GPU_DATA state_delta_gpu;
    GPU_DATA gate_gpu;
    GPU_DATA gate_delta_gpu;
    GPU_DATA save_gpu;
    GPU_DATA save_delta_gpu;
    GPU_DATA concat_gpu;
    GPU_DATA concat_delta_gpu;

    GPU_DATA binary_input_gpu;
    GPU_DATA binary_weights_gpu;

    GPU_DATA mean_gpu;
    GPU_DATA variance_gpu;

    GPU_DATA rolling_mean_gpu;
    GPU_DATA rolling_variance_gpu;

    GPU_DATA variance_delta_gpu;
    GPU_DATA mean_delta_gpu;

    GPU_DATA x_gpu;
    GPU_DATA x_norm_gpu;
    GPU_DATA weights_gpu;
    GPU_DATA weight_updates_gpu;

    GPU_DATA biases_gpu;
    GPU_DATA bias_updates_gpu;

    GPU_DATA scales_gpu;
    GPU_DATA scale_updates_gpu;

    GPU_DATA output_gpu;
    GPU_DATA delta_gpu;
    GPU_DATA rand_gpu;
    GPU_DATA squared_gpu;
    GPU_DATA norms_gpu;
    
    #ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
    #endif
    #endif
};

void free_layer(layer);

#endif
