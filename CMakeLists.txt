cmake_minimum_required(VERSION 3.7.2)
project(darknet)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(DARKNET_CPU "Enable cpu support" ON)
option(DARKNET_OPENCV "Enable OpenCV support" OFF)
option(DARKNET_CUDA "Enable Cuda support" OFF)
option(DARKNET_OPENCL "Enable OpenCL support")
option(DARKNET_TESTING "Build the unit tests" OFF)
option(DARKNET_CUDNN "Enable Cudnn support" OFF)
option(DARKNET_CUDA_MEM_DEBUG "Enable Cuda memory debug" OFF)


set (DARKNET_INCLUDE_DIRS "")
set (DARKNET_LINK_DIRS "")
set (DARKNET_LIBRARIES "m" "pthread")
set (DARKNET_EXE "darknet")

if (DARKNET_OPENCV)
    find_package(OpenCV REQUIRED)
    list (APPEND DARKNET_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})
    list (APPEND DARKNET_LIBRARIES ${OpenCV_LIBS})
    list (APPEND DARKNET_LINK_DIRS ${OpenCV_LINK_DIRECTORIES})
endif()

if (DARKNET_OPENCL)
    set (DARKNET_CPU OFF)
    find_package(OpenCL REQUIRED)
    find_package(clBLAS REQUIRED)
    list (APPEND DARKNET_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} ${CLBLAS_INCLUDE_DIRS})
    list (APPEND DARKNET_LIBRARIES ${OpenCL_LIBRARIES} ${CLBLAS_LIBRARIES})
    #list (APPEND DARKNET_LINK_DIRS ${OpenCV_LINK_DIRECTORIES})
endif()

if (DARKNET_UNIT)
    set (DARKNET_CPU OFF)
    find_package(Catch REQUIRED)
    list (APPEND DARKNET_INCLUDE_DIRS ${Catch_INCLUDE_DIRS})
endif()

#if (DARKNET_CUDA)
#    find_package(CUDA REQUIRED)
#    list (APPEND DARKNET_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
#    list (APPEND DARKNET_LIBRARIES ${CUDA_LIBRARIES})
#    list (APPEND DARKNET_LINK_DIRS ${CUDA_LINK_DIRECTORIES})
#endif()
#
#if (DARKNET_CUDNN)
#    find_package(CUDNN)
#    find_package(CUDA REQUIRED)
#    list (APPEND DARKNET_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS})
#    list (APPEND DARKNET_LIBRARIES ${CUDNN_LIBRARIES})
#    list (APPEND DARKNET_LINK_DIRS ${CUDNN_LINK_DIRECTORIES})
#endif()

set (DARKNET_SOURCES
    "src/activation_layer.c"
    "src/activation_layer.h"
    "src/activations.c"
    "src/activations.h"
    "src/art.c"
    "src/avgpool_layer.c"
    "src/avgpool_layer.h"
    "src/batchnorm_layer.c"
    "src/batchnorm_layer.h"
    "src/blas.c"
    "src/blas.h"
    "src/box.c"
    "src/box.h"
    "src/bugfixes.h"
    "src/captcha.c"
    "src/cifar.c"
    "src/classifier.c"
    "src/classifier.h"
    "src/coco.c"
    "src/col2im.c"
    "src/col2im.h"
    "src/compare.c"
    "src/connected_layer.c"
    "src/connected_layer.h"
    "src/convolutional_layer.c"
    "src/convolutional_layer.h"
    "src/cost_layer.c"
    "src/cost_layer.h"
    "src/crnn_layer.c"
    "src/crnn_layer.h"
    "src/crop_layer.c"
    "src/crop_layer.h"
    "src/cuda.h"
    "src/data.c"
    "src/data.h"
    "src/deconvolutional_layer.c"
    "src/deconvolutional_layer.h"
    "src/demo.c"
    "src/demo.h"
    "src/detection_layer.c"
    "src/detection_layer.h"
    "src/detector.c"
    "src/dice.c"
    "src/dropout_layer.c"
    "src/dropout_layer.h"
    "src/gemm.c"
    "src/gemm.h"
    "src/go.c"
    "src/gru_layer.c"
    "src/gru_layer.h"
    "src/im2col.c"
    "src/im2col.h"
    "src/image.c"
    "src/image.h"
    "src/layer.c"
    "src/layer.h"
    "src/list.c"
    "src/list.h"
    "src/local_layer.c"
    "src/local_layer.h"
    "src/lsd.c"
    "src/matrix.c"
    "src/matrix.h"
    "src/maxpool_layer.c"
    "src/maxpool_layer.h"
    "src/network.c"
    "src/network.h"
    "src/nightmare.c"
    "src/normalization_layer.c"
    "src/normalization_layer.h"
    "src/option_list.c"
    "src/option_list.h"
    "src/parser.c"
    "src/parser.h"
    "src/region_layer.c"
    "src/region_layer.h"
    "src/regressor.c"
    "src/reorg_layer.c"
    "src/reorg_layer.h"
    "src/rnn.c"
    "src/rnn_layer.c"
    "src/rnn_layer.h"
    "src/rnn_vid.c"
    "src/route_layer.c"
    "src/route_layer.h"
    "src/shortcut_layer.c"
    "src/shortcut_layer.h"
    "src/softmax_layer.c"
    "src/softmax_layer.h"
    "src/stb_image.h"
    "src/stb_image_write.h"
    "src/super.c"
    "src/swag.c"
    "src/tag.c"
    "src/tree.c"
    "src/tree.h"
    "src/unit_network.log"
    "src/utils.c"
    "src/utils.h"
    "src/voxel.c"
    "src/win32"
    "src/writing.c"
    "src/yolo.c"
)

set (DARKNET_MAIN_SOURCE
    "src/darknet.c"
)

set (DARKNET_CPU_SOURCES
    "src/cpu.c"
)

set (DARKNET_OPENCL_SOURCES
    "src/activation_kernels.c"
    "src/activation_kernels.cl"
    "src/avgpool_layer_kernels.c"
    "src/avgpool_layer_kernels.cl"
    "src/blas_kernels.c"
    "src/blas_kernels.cl"
    "src/col2im_kernels.c"
    "src/col2im_kernels.cl"
    "src/convolutional_kernels.c"
    "src/convolutional_kernels.cl"
    "src/crop_layer_kernels.c"
    "src/crop_layer_kernels.cl"
    "src/deconvolutional_kernels.c"
    "src/dropout_layer_kernels.c"
    "src/dropout_layer_kernels.cl"
    "src/im2col_kernels.c"
    "src/im2col_kernels.cl"
    "src/maxpool_layer_kernels.c"
    "src/maxpool_layer_kernels.cl"
    "src/network_kernels.c"
    "src/opencl.c"
)

set (DARKNET_CUDA_SOURCES
    "src/activation_kernels.cu"
    "src/avgpool_layer_kernels.cu"
    "src/blas_kernels.cu"
    "src/col2im_kernels.cu"
    "src/convolutional_kernels.cu"
    "src/crop_layer_kernels.cu"
    "src/deconvolutional_kernels.cu"
    "src/dropout_layer_kernels.cu"
    "src/im2col_kernels.cu"
    "src/maxpool_layer_kernels.cu"
    "src/network_kernels.cu"
    "src/softmax_layer_kernels.cu"
    "src/cuda.c"
)

set (DARKNET_UNIT_SOURCES
    "src/unit.c"
    "src/unit.h"
    "src/blas_unit.c"
    "src/col2im_unit.c"
    "src/convolutional_unit.c"
    "src/gemm_unit.c"
    "src/maxpool_unit.c"
    "src/network_unit.c"
    "src/region_unit.c"
)

if (DARKNET_OPENCL)

    add_executable(${DARKNET_EXE} ${DARKNET_SOURCES}
        ${DARKNET_OPENCL_SOURCES} ${DARKNET_MAIN_SOURCE})
    target_compile_definitions(${DARKNET_EXE} PUBLIC GPU=1 OPENCL=1)

elseif(DARKNET_CPU)

    add_executable(${DARKNET_EXE} ${DARKNET_SOURCES}
        ${DARKNET_CPU_SOURCES} ${DARKNET_MAIN_SOURCE})

endif()

target_link_libraries(${DARKNET_EXE} PUBLIC ${DARKNET_LIBRARIES})

if (DARKNET_OPENCL)
    
endif()
