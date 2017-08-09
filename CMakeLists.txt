cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)

project(darknet LANGUAGES C CXX)

option(DARKNET_CPU "Enable cpu support" ON)
option(DARKNET_OPENCV "Enable OpenCV support" OFF)
option(DARKNET_CUDA "Enable Cuda support" OFF)
option(DARKNET_OPENCL "Enable OpenCL support" OFF)
option(DARKNET_TESTING "Build the unit tests" OFF)
option(DARKNET_CUDNN "Enable Cudnn support" OFF)
option(DARKNET_CUDA_MEM_DEBUG "Enable Cuda memory debug" OFF)
option(DARKNET_DEBUG "Enable debug mode" OFF)
option(DARKNET_STATIC "Build static library" ON)
option(DARKNET_SHARED "Build shared library" ON)

include(GNUInstallDirs)

set (DARKNET_DEFINITIONS "")
set (DARKNET_BINARY bindarknet)
set (DARKNET_LIBRARY libdarknet)
set (DARKNET_LIBRARY_STATIC libdarknet_s)
set (DARKNET_CFLAGS "-Wall" "-Wno-misleading-indentation" "-Wfatal-errors")

if (DARKNET_DEBUG OR DARKNET_TESTING)
    list (APPEND DARKNET_CFLAGS "-O0 -g")
else()
    list (APPEND DARKNET_CFLAGS "-Ofast")
endif()

# Find OpenCV
if (DARKNET_OPENCV)
	find_package(OpenCV REQUIRED)
	add_library(opencv INTERFACE IMPORTED)
	set_property(TARGET opencv PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
    set_property(TARGET opencv PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCV_LIBRARIES})
    list (APPEND DARKNET_DEFINITIONS OPENCV)
endif()

# Find OpenCL
if (DARKNET_OPENCL)
	find_package(OpenCL REQUIRED)
    find_package(clBLAS REQUIRED)
    add_library(opencl INTERFACE IMPORTED)
    add_library(clblas INTERFACE IMPORTED)
    set_property(TARGET opencl PROPERTY
    	INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
    set_property(TARGET clblas PROPERTY
    	INTERFACE_INCLUDE_DIRECTORIES ${CLBLAS_INCLUDE_DIRS})
    set_property(TARGET opencl PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})
    set_property(TARGET clblas PROPERTY
        INTERFACE_LINK_LIBRARIES ${CLBLAS_LIBRARIES})
    list (APPEND DARKNET_DEFINITIONS GPU OPENCL)
endif()

# Find Catch
if (DARKNET_UNIT)
	set (DARKNET_CPU OFF)
	find_package(Catch REQUIRED)
	add_library(catch INTERFACE IMPORTED)
	set_property(TARGET catch PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${Catch_INCLUDE_DIRS})
endif()

set (DARKNET_SOURCES
	"src/activation_layer.c"
    "src/activations.c"
    "src/art.c"
    "src/avgpool_layer.c"
    "src/batchnorm_layer.c"
    "src/blas.c"
    "src/box.c"
    "src/captcha.c"
    "src/cifar.c"
    "src/classifier.c"
    "src/coco.c"
    "src/col2im.c"
    "src/compare.c"
    "src/connected_layer.c"
    "src/convolutional_layer.c"
    "src/cost_layer.c"
    "src/crnn_layer.c"
    "src/crop_layer.c"
    "src/data.c"
    "src/deconvolutional_layer.c"
    "src/demo.c"
    "src/detection_layer.c"
    "src/detector.c"
    "src/dice.c"
    "src/dropout_layer.c"
    "src/gemm.c"
    "src/go.c"
    "src/gru_layer.c"
    "src/im2col.c"
    "src/image.c"
    "src/layer.c"
    "src/list.c"
    "src/local_layer.c"
    "src/lsd.c"
    "src/matrix.c"
    "src/maxpool_layer.c"
    "src/network.c"
    "src/nightmare.c"
    "src/normalization_layer.c"
    "src/option_list.c"
    "src/parser.c"
    "src/region_layer.c"
    "src/regressor.c"
    "src/reorg_layer.c"
    "src/rnn.c"
    "src/rnn_layer.c"
    "src/rnn_vid.c"
    "src/route_layer.c"
    "src/shortcut_layer.c"
    "src/softmax_layer.c"
    "src/super.c"
    "src/swag.c"
    "src/tag.c"
    "src/tree.c"
    "src/unit_network.log"
    "src/utils.c"
    "src/voxel.c"
    "src/win32"
    "src/writing.c"
    "src/yolo.c"
)

set (DARKNET_PUBLIC_HEADERS
    "src/activation_layer.h"
    "src/activations.h"
    "src/avgpool_layer.h"
    "src/batchnorm_layer.h"
    "src/blas.h"
    "src/box.h"
    "src/bugfixes.h"
    "src/classifier.h"
    "src/col2im.h"
    "src/connected_layer.h"
    "src/convolutional_layer.h"
    "src/cost_layer.h"
    "src/crnn_layer.h"
    "src/crop_layer.h"
    "src/cuda.h"
    "src/data.h"
    "src/deconvolutional_layer.h"
    "src/demo.h"
    "src/detection_layer.h"
    "src/dropout_layer.h"
    "src/gemm.h"
    "src/gru_layer.h"
    "src/im2col.h"
    "src/image.h"
    "src/layer.h"
    "src/list.h"
    "src/local_layer.h"
    "src/matrix.h"
    "src/maxpool_layer.h"
    "src/network.h"
    "src/normalization_layer.h"
    "src/option_list.h"
    "src/parser.h"
    "src/region_layer.h"
    "src/reorg_layer.h"
    "src/rnn_layer.h"
    "src/route_layer.h"
    "src/shortcut_layer.h"
    "src/softmax_layer.h"
    "src/stb_image.h"
    "src/stb_image_write.h"
    "src/tree.h"
    "src/unit.h"
    "src/utils.h"
)

if (WIN32)
    set (DARKNET_WINDOWS_SOURCES
        "src/win32/pthread.cpp"
    )

    set (DARKNET_WINDOWS_PUBLIC_HEADERS
        "src/win32/pthread.h"
    )
endif()

set (DARKNET_MAIN_SOURCE
    "src/darknet.c"
)

if (DARKNET_CPU)
	set (DARKNET_CPU_SOURCES
    	"src/cpu.c"
	)
endif()

if (DARKNET_OPENCL)
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
endif()

if (DARKNET_CUDA)
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
endif()

if (DARKNET_UNIT)
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
endif()

function (ConfigureBinary target)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "darknet")    
    target_compile_definitions(${target} PUBLIC ${DARKNET_DEFINITIONS})
    target_compile_options(${target} PRIVATE ${DARKNET_CFLAGS})

    if (DARKNET_OPENCL)
        target_link_libraries(${target} opencl clblas)
    endif()

    if (DARKNET_OPENCV)
        target_link_libraries(${target} opencv)
    endif()

    if (DARKNET_TESTING)
        target_link_libraries(${target} catch)
    endif()

    target_link_libraries(${target} "m" "pthread")
endfunction()

function (ConfigureLibrary target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    install(TARGETS ${target} EXPORT ${CMAKE_PROJECT_NAME}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if (DARKNET_SHARED)
    add_library(${DARKNET_LIBRARY} SHARED ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
	   ${DARKNET_OPENCL_SOURCES} ${DARKNET_CUDA_SOURCES} ${DARKNET_WINDOWS_SOURCES})
    ConfigureBinary(${DARKNET_LIBRARY})
    ConfigureLibrary(${DARKNET_LIBRARY})
endif()

if (DARKNET_STATIC)
    add_library(${DARKNET_LIBRARY_STATIC} STATIC ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
       ${DARKNET_OPENCL_SOURCES} ${DARKNET_CUDA_SOURCES} ${DARKNET_WINDOWS_SOURCES})
    ConfigureBinary(${DARKNET_LIBRARY_STATIC})
    ConfigureLibrary(${DARKNET_LIBRARY_STATIC})
endif()

add_executable(${DARKNET_BINARY} ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
    ${DARKNET_OPENCL_SOURCES} ${DARKNET_CUDA_SOURCES} ${DARKNET_WINDOWS_SOURCES}
    ${DARKNET_MAIN_SOURCE})

ConfigureBinary(${DARKNET_BINARY})

install(FILES ${DARKNET_PUBLIC_HEADERS} ${DARKNET_WINDOWS_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME})

install(EXPORT ${CMAKE_PROJECT_NAME} DESTINATION share/${CMAKE_PROJECT_NAME}/cmake)

export(TARGETS ${DARKNET_LIBRARY} FILE darknetConfig.cmake)
