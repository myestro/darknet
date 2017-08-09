#ifndef DARKNET_UNIT_TEST_
#define DARKNET_UNIT_TEST_

#include "layer.h"
#include "network.h"

static char* yolo_configuration_file = "cfg/yolo.cfg";
static char* yolo_weights_file = "weights/yolo.weights";
static char* coco_names_file = "data/coco.names";
static char* coco_data_file = "cfg/coco.data";

void fillRandom(float *array, const size_t size);

void compare_array(const float *a, const float *b,
	const size_t size, const float threshold);

layer* getLayer(network net, LAYER_TYPE type);

#endif
