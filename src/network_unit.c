#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "convolutional_layer.h"
#include "cuda.h"
#include "unit.h"
#include "parser.h"
#include "im2col.h"
#include "blas.h"
#include "gemm.h"
#include "box.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "option_list.h"
#include "utils.h"

TEST_CASE("Opencl kernel bash", "[network][opencl]")
{
	const float threshold = 0.9;
	char *imagePath = "test.png";
	char *outputCPU = "output_cpu.png";
	char *outputGPU = "output_gpu.png";
	char *datacfg = "coco.data";
	char *cfgfile = "yolo.cfg";
	char *weightsfile = "yolo.weights";
	char *name_list = "coco.names";
	int print = 0;

	opencl_init(NULL, NULL, NULL);

	network net = load_network(cfgfile, weightsfile, 0);

	network_state state;
	state.net = net;
	state.index = 0;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	state.workspace = net.workspace;
	state.workspace_gpu = net.workspace_gpu;

	const size_t testSize = net.layers[0].batch * net.layers[0].outputs;

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

	SECTION("Complete darknet test")
	{
		for (size_t i = 0; i < net.n; ++i)
		{
			state.index = i;
			layer l = net.layers[i];
			if (l.delta_gpu)
			{
				fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
				fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
			}

			gpu_index = -1;
			l.forward(l, state);

			if (l.type == REGION)
				memcpy(l.output, B, l.outputs * l.batch * sizeof(float));

			gpu_index = 1;
			l.forward_gpu(l, state);

			printf("Layer: %ld %s\n", i, get_layer_string((LAYER_TYPE) l.type));
			cuda_pull_array(l.output_gpu, A, l.outputs*l.batch);

			if (l.type == REGION)
				compare_array(l.output, B, l.outputs*l.batch, threshold, print);
			else
				compare_array(l.output, A, l.outputs*l.batch, threshold, print);
			
			state.input = l.output;
			state.input_gpu = l.output_gpu;

			if(l.truth) state.truth_gpu = l.output_gpu;
		}
	}

	SECTION("layer test")
	{
		for (size_t i = 0; i < net.n; ++i)
		{
			state.index = i;
			layer l = net.layers[i];

			gpu_index = -1;
			l.forward(l, state);

			if (l.type == REGION)
				memcpy(C, l.output, l.outputs * l.batch * sizeof(float));

			gpu_index = 1;
			l.forward_gpu(l, state);

			printf("Layer: %ld %s\n", i, get_layer_string((LAYER_TYPE) l.type));
			cuda_pull_array(l.output_gpu, B, l.outputs*l.batch);

			if (l.type == REGION)
				compare_array(B, C, l.outputs*l.batch, threshold, 0);
			else
				compare_array(l.output, B, l.outputs*l.batch, threshold, 0);
		}
	}

	SECTION("network with real input")
	{
		set_batch_network(&net, 1);

		layer l = net.layers[net.n - 1];
		char *outputName = 0;

//		list *options = read_data_cfg(datacfg);
//    	char *name_list = option_find_str(options, "names", "data/names.list");

		image **alphabet = load_alphabet();
		char **names = get_labels(name_list);

		image im = load_image_color(imagePath, 0, 0);
		image sized = letterbox_image(im, net.w, net.h);

		box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
        for(size_t j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float));

        float *X = sized.data;
    	float thresh = 0.25;
    	float hier_thresh = 0.5;
    	float nms = 0.4;


		SECTION("network with real input CPU")
		{
			gpu_index = -1;
			network_predict(net, X);
			outputName = outputCPU;
		}

		SECTION("network with real input GPU")
		{
			gpu_index = 1;
			network_predict(net, X);
			outputName = outputGPU;
		}

		get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(sized, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        save_image(sized, outputName);

		free_image(im);
		free_image(sized);
		free(boxes);
		free_ptrs((void **)probs, l.w * l.h * l.n);

	}

	cuda_free(A_gpu);
	opencl_deinit();
	free_network(net);
	free(A);
	free(B);
	free(C);
}