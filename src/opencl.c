#ifdef GPU
#ifdef OPENCL

#include "cuda.h"
int gpu_index = 0;

cl_context opencl_context;
cl_command_queue opencl_queue;
cl_device_id opencl_device;
cl_bool opencl_foreign_context;

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include "blas.h"
#include "utils.h"

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

EXTERN void activation_kernel_init(void);
EXTERN void blas_kernel_init(void);
EXTERN void col2im_kernel_init(void);
EXTERN void convolutional_kernel_init(void);
EXTERN void im2col_kernel_init(void);
EXTERN void maxpool_kernel_init(void);
EXTERN void gemm_kernel_init(void);
EXTERN void avgpool_kernel_init(void);
EXTERN void crop_kernel_init(void);
EXTERN void dropout_kernel_init(void);

EXTERN void activation_kernel_release(void);
EXTERN void blas_kernel_release(void);
EXTERN void col2im_kernel_release(void);
EXTERN void convolutional_kernel_release(void);
EXTERN void im2col_kernel_release(void);
EXTERN void maxpool_kernel_release(void);
EXTERN void gemm_kernel_release(void);
EXTERN void avgpool_kernel_release(void);
EXTERN void crop_kernel_release(void);
EXTERN void dropout_kernel_release(void);

#undef EXTERN


dim3 dim3_create(const size_t x, const size_t y, const size_t z)
{
	dim3 retVal;
	retVal.x = x;
	retVal.y = y;
	retVal.z = z;

	return retVal;
}

void opencl_load(const char *fileName, cl_program *output)
{
	FILE *fp;
	size_t lSize, readSize;
	char * sourceBuffer;

	fp = fopen(fileName, "r");
	
	if (fp == NULL)
	{
		printf("opencl_load: Could not open file: %s\n", fileName);
		fclose(fp);
		return;
	}

	// Determine file size.
	fseek(fp, 0, SEEK_END);
	lSize = ftell(fp);
	rewind(fp);

	sourceBuffer = (char*) malloc(sizeof(char) * lSize);

	if (sourceBuffer == NULL)
	{
		printf("opencl_load: Could not allocate memory for file: %s\n",
			fileName);
		fclose(fp);
		return;
	}

	readSize = fread(sourceBuffer, 1, lSize, fp);
	fclose(fp);

	if (readSize > lSize)
	{
		printf("opencl_load: failed to read file: %s\n", fileName);
		free(sourceBuffer);
		return;
	}

	opencl_load_buffer(sourceBuffer, readSize, output);

	free(sourceBuffer);
}

void opencl_load_buffer(const char *buffer, const size_t size, cl_program *output)
{
	cl_int clErr;

	*output = clCreateProgramWithSource(opencl_context, CL_TRUE,
		(const char**)&buffer, &size, &clErr);

	if (clErr != CL_SUCCESS)
	{
		printf("opencl_load: Could not create program. Error code %d.\n", clErr);
		return;
	}

	clErr = clBuildProgram(*output, CL_TRUE, &opencl_device,
		NULL, NULL, NULL);

	if (clErr != CL_SUCCESS)
	{
		printf("opencl_load: Could not compile. Error code: %d\n", clErr);
		size_t len;
		char *buffer = (char*)calloc(0x10000000, sizeof(char));
		clGetProgramBuildInfo(*output, opencl_device, CL_PROGRAM_BUILD_LOG, 0x10000000 * sizeof(char), buffer, &len);
		printf("CL_PROGRAM_BUILD_LOG:\n%s\n", buffer);
		free(buffer);
	}
}

void opencl_create_kernel(cl_program *program, const char *kernelName,
	cl_kernel *kernel)
{
	cl_int clErr;

	*kernel = clCreateKernel(*program, kernelName, &clErr);

	if (clErr)
	{
		printf("opencl_create_kernel: Could not create kernel %s.\n",
			kernelName);
	}
}

void opencl_init(cl_context context, cl_command_queue queue,
	cl_device_id device)
{
	if (opencl_context != NULL)
		opencl_deinit();

	cl_int clErr;

	if (context != NULL && queue != NULL && device != NULL)
	{
		// Use foreign OpenCL context.
		opencl_context = context;
		opencl_queue = queue;
		opencl_device = device;
		opencl_foreign_context = CL_TRUE;
	}
	else
	{
		// Create OpenCL context from scratch.
		cl_platform_id clPlatform = 0;
		cl_uint clNumPlatforms = 0;
		cl_uint clNumDevices = 0;
		cl_context_properties clProps[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

		clErr = clGetPlatformIDs(CL_TRUE, &clPlatform, &clNumPlatforms);
		
		if (clErr != CL_SUCCESS)
		{
			printf("opencl_init: Could not get platform IDs.\n");
			return;
		}

		clErr = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, CL_TRUE,
			&opencl_device, &clNumDevices);
		
		if (clErr != CL_SUCCESS)
		{
			printf("opencl_init: Could not get device IDs.\n");
			return;
		}

		clProps[1] = (cl_context_properties) clPlatform;

		opencl_context = clCreateContext(clProps, CL_TRUE,
			&opencl_device, NULL, NULL, &clErr);
		
		if (clErr != CL_SUCCESS)
		{
			printf("opencl_init: Could not create context.\n");
			return;
		}

		opencl_queue = clCreateCommandQueue(opencl_context,
			opencl_device, CL_FALSE, &clErr);
		
		if (clErr != CL_SUCCESS)
		{
			printf("opencl_init: Could not create queue.\n");
			return;
		}

		opencl_foreign_context = CL_FALSE;

#if defined(DARKNET_VERBOSE_OPENCL)
		// Print out usefull information.
		const size_t bufferSize = 2048;
		char *buffer = (char*)calloc(bufferSize, sizeof(char));

		clGetDeviceInfo(opencl_device, CL_DEVICE_NAME, bufferSize * sizeof(char), buffer, NULL);
		printf("Device name: %s\n", buffer);
		clGetDeviceInfo(opencl_device, CL_DEVICE_VENDOR, bufferSize * sizeof(char), buffer, NULL);
		printf("Device vendor: %s\n", buffer);
		clGetDeviceInfo(opencl_device, CL_DEVICE_VERSION, bufferSize * sizeof(char), buffer, NULL);
		printf("Device opencl availability: %s\n", buffer);
		clGetDeviceInfo(opencl_device, CL_DRIVER_VERSION, bufferSize * sizeof(char), buffer, NULL);
		printf("Device opencl used: %s\n", buffer);
		free(buffer);
#endif
	}

	activation_kernel_init();
	blas_kernel_init();
	col2im_kernel_init();
	convolutional_kernel_init();
	im2col_kernel_init();
	maxpool_kernel_init();
	gemm_kernel_init();
	avgpool_kernel_init();
	crop_kernel_init();
	dropout_kernel_init();

	// Activate darknet GPU path.
	gpu_index = 1;
}

void opencl_deinit()
{
	if (opencl_context == NULL)
		return;

	activation_kernel_release();
	blas_kernel_release();
	col2im_kernel_release();
	convolutional_kernel_release();
	im2col_kernel_release();
	maxpool_kernel_release();
	gemm_kernel_release();
	avgpool_kernel_release();
	crop_kernel_release();
	dropout_kernel_release();

	clFinish(opencl_queue);
	gpu_index = -1;

	if (opencl_foreign_context)
	{
		opencl_context = 0;
		opencl_queue = 0;
		return;
	}
	
	clReleaseCommandQueue(opencl_queue);
	clReleaseContext(opencl_context);

	opencl_queue = 0;
	opencl_context = 0;
}

void opencl_kernel(cl_kernel kernel, const dim3 globalItemSize,
	const dim3 localItemSize, const int argc, ...)
{
	cl_int clErr;

	va_list vl;
	va_start(vl, argc);

	size_t argSize = 0;
	void *argValue = NULL;

	for (int i = 0, j = 0; i < argc; i+=2, ++j)
	{
		argValue = va_arg(vl, void*);
		argSize = va_arg(vl, size_t);

		clErr = clSetKernelArg(kernel, j, argSize, argValue);
		
		if (clErr != CL_SUCCESS)
		{
			const size_t bufferSize = 2048;
			char *kernelName = (char*) calloc(bufferSize, sizeof(char));
		
			clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
			printf("opencl_kernel %s could not set kernel argument. Errorcode: %d\n", kernelName, clErr);

			free(kernelName);
		}
	}

	va_end(vl);

	size_t globalItems[3];//, localItems[3];
	globalItems[0] = globalItemSize.x;
	globalItems[1] = globalItemSize.y;
	globalItems[2] = globalItemSize.z;
	//localItems[0] = localItemSize.x;
	//localItems[1] = localItemSize.y;
	//localItems[2] = localItemSize.z;

	clErr = clEnqueueNDRangeKernel(opencl_queue, kernel, 3, NULL, globalItems,
		NULL, 0, NULL, NULL);

	if (clErr != CL_SUCCESS)
	{
		const size_t bufferSize = 2048;
		char *kernelName = (char*) calloc(bufferSize, sizeof(char));
		
		clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
		printf("opencl %s error code: %d\n", kernelName, clErr);

		free(kernelName);
	}
}

cl_mem cuda_make_array(float *x, size_t n)
{
	cl_mem buffer = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE,
		n * sizeof(cl_float), NULL, NULL);

	if (x != NULL)
		cuda_push_array(buffer, x, n);
	else
	{
		float *cptr = (float*) calloc(n * sizeof(float), 1);
		if (cptr != NULL)
			cuda_push_array(buffer, cptr, n);
		free(cptr);
	}

	return buffer;
}

cl_mem cuda_make_int_array(size_t n)
{
	return clCreateBuffer(opencl_context, CL_MEM_READ_WRITE,
		n * sizeof(cl_int), NULL, NULL);	
}

void cuda_push_array(cl_mem x_gpu, float *x, size_t n)
{
	cl_int clErr = clEnqueueWriteBuffer(opencl_queue, x_gpu, CL_TRUE, 0,
		n * sizeof(cl_float), x, 0, NULL, NULL);

	if (clErr != CL_SUCCESS)
		printf("Could not push array to device. Error code %d\n", clErr);
}

void cuda_pull_array(cl_mem x_gpu, float *x, size_t n)
{
	cl_int clErr = clEnqueueReadBuffer(opencl_queue, x_gpu, CL_TRUE, 0,
		n * sizeof(cl_float), x, 0, NULL, NULL);

	if (clErr != CL_SUCCESS)
		printf("Could not pull array from device. Error code %d\n", clErr);
}

void cuda_set_device(int n)
{
	// Not available under OpenCL.
	return;
}

void cuda_free(cl_mem x_gpu)
{
	clReleaseMemObject(x_gpu);
}

void cuda_random(cl_mem x_gpu, size_t n)
{
	//opencl_kernel(OPENCL_RANDOM_KERNEL, n, BLOCK, 1, x_gpu, sizeof(x_gpu));
}

float cuda_compare(cl_mem x_gpu, float *x, size_t n, char *s)
{
	float *tmp = (float*)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

dim3 cuda_gridsize(size_t n)
{
	return dim3_create(n, 1, 1);
	size_t maxWorkGroupSize = 0;
	clGetDeviceInfo(opencl_device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(size_t), &maxWorkGroupSize, NULL);

	size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
	if (x > maxWorkGroupSize)
    {
        x = ceil(sqrt((long double)k));
        y = (n-1)/(x*BLOCK) + 1;
    }
	
	dim3 ret = dim3_create(x, y, 1);
	
    return ret;
}

float cuda_mag_array(cl_mem x_gpu, size_t n)
{
	float *temp = (float*) calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);

    if(temp) free(temp);

    return m;
}

void cuda_dump_mem_stat()
{
	size_t used, total;

	clGetDeviceInfo(opencl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
		sizeof(size_t), &total, NULL);

	clGetDeviceInfo(opencl_device, CL_DEVICE_LOCAL_MEM_SIZE,
		sizeof(size_t), &used, NULL);

	printf("OpenCL memory status: Free/Total = [%lu]/[%lu]\n", total - used, total);
}

#endif // OPENCL
#endif // GPU