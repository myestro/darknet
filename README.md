![Darknet Logo](https://github.com/prabindh/darknet/blob/master/arapaho/darknetcpplogo.png)

# Darknet-cpp-opencl

*Darknet-cpp-opencl project is a fork of the bug-fixed and C++ compilable version of darknet, an open source neural network framework written in C, CUDA and now with OpenCL support. Darknet-cpp-opencl builds and is tested on Linux. It should also work on Windows and Mac, but is still untested.*

## Features

*Uses same code-base as original darknet (ie same .c files are used). Modification is done only for runtime bug-fixes, compile time fixes for c++, and the build system itself. The OpenCL support is realised by adding new .c and .cl files which complement the cuda and cpu implementation.*


*The Linux build system supports 3 targets*

 * original darknet (with gcc compiler)
 * darknet-cpp (with g++ compiler and Visual Studio compiler)
 * Shared library (libdarknet-cpp-shared.so)

*The Linux build system also supports 3 computation backends*

 * The original cpu implementation
 * The original cuda implementation
 * The new OpenCL implementation

For questions about this port, please use the [Google group](https://groups.google.com/forum/#!forum/darknet-opencl-port)

## Requirements

The new OpenCL backend needs the AMD clBLAS libraries as a replacement for the cuda BLAS libraries. You can find these libraries [here](https://github.com/clMathLibraries/clBLAS). Also you should install your OpenCL drivers.

Also catch is needed for the unit testing and can be found here - [catch](https://github.com/philsquared/Catch)

On Ubuntu you can run the following apt-get command

 * `sudo apt-get install libclblas2 clblas-client libclblas-dev catch`

For the OpenCL drivers please follow the instructions given by your OpenCL vendor

 * [AMD](http://developer.amd.com/tools-and-sdks/opencl-zone/)
 * [INTEL](https://software.intel.com/en-us/intel-opencl)
 * [NVIDIA](https://developer.nvidia.com/opencl)

## Usage

Using the Makefile in the root directory of the darknet source repository,

 * `make darknet` - darknet, without any GPU support.
 * `make darknet GPU=1 CUDA=1` - darknet, with cuda support.
 * `make darknet GPU=1 OPENCL=1` - darknet, with OpenCL support.
 * `make darknet-unit GPU=1 GPU_UNIT=1 OPENCL=1` - Builds the cpu/gpu comparison unit tests.
 * `make darknet-unit GPU=1 GPU_UNIT=1 CUDA=1` - Same as above but with cuda support.
 * `make darknet-cpp` - The cpp fixes from the darknet-cpp port are also available.

Please begin by runnning darknet-unit and verifying that your opencl platform is available.

### CMake

** Caution CMake support is experimental **

You can also use cmake to generate a Makefile. Be warned just running the cmake command will replace the hand crafted makefile. Instead use the following commands to generate a makefile:
 * `mkdir build` - Create a build folder
 * `cmake -Bbuild -H. -DDARKNET_OPENCL:BOOL=ON` - Generate the makefiles
 * `cmake --build build` - Compile the darknet sources
 * `cp build/darknet .` - Place the binary where it should be
 * `./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg` - Try it out
 
## Training the darknet with opencl support

**Training a network should be identical to the process described [here](https://pjreddie.com/darknet/yolo/). But this is still untested. **

# How to file issues

For issues with the darknet-cpp port, use the link - [darknet-cpp](https://github.com/prabindh/darknet/issues.)

For issues with the darknet-cpp-opencl port, use the link -  [darknet-cpp-opencl](https://groups.google.com/forum/#!forum/darknet-opencl-port)

For general issues with the darknet use the original mailing list - [darknet](https://groups.google.com/forum/#!forum/darknet)

Information required for filing an issue:

  * Options enabled in Makefile
  * Did the unit test run?
  * Platform being used (OS version, GPU type, OpenCL version, and OpenCV version)

# Darknet-cpp for Windows #

**I have not tested this on windows yet, but it should still work.**

The solution file requires the below repository.

https://github.com/prabindh/darknet-cpp-windows

The Windows port does not require any additional downloads (like pthreads), and builds the same darknet code-base for Windows, to generate the darknet.dll. Building the Arapaho C++ API and test wrapper, creates arapaho.exe, that works exactly the same way as arapaho on Linux.

# Darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
