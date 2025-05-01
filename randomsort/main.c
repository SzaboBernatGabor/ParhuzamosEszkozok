#include <stdio.h>
#define CL_TARGET_OPENCL_VERSION 220
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include "kernel_loader.h"
#include <string.h>

#define ARRAY_SIZE 12
#define NUM_THREADS 1024

int main() {
    int data[ARRAY_SIZE];
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        data[i] = rand() % 100;
    }

    printf("Eredeti tömb:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_mem input_mem, result_flag;

    int success = 0;
    cl_int ret;

    int error_code;
    char* source_str = load_kernel_source("randomsort.cl", &error_code);
    if (error_code != 0) {
        fprintf(stderr, "Nem sikerült betölteni a kernelt!\n");
        return 1;
    }
    const size_t source_size = strlen(source_str);

    ret = clGetPlatformIDs(1, &platform_id, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs hiba: %d\n", ret);
        return 1;
    }

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs hiba: %d\n", ret);
        return 1;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext hiba: %d\n", ret);
        return 1;
    }

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue hiba: %d\n", ret);
        return 1;
    }

    input_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * ARRAY_SIZE, data, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer (input_mem) hiba: %d\n", ret);
        return 1;
    }

    result_flag = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer (result_flag) hiba: %d\n", ret);
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource hiba: %d\n", ret);
        return 1;
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        fprintf(stderr, "clBuildProgram hiba: %d\n", ret);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);

        return 1;
    }

    kernel = clCreateKernel(program, "random_sort", &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel hiba: %d\n", ret);
        return 1;
    }

    int array_size = ARRAY_SIZE;

    int zero = 0;
    ret = clEnqueueWriteBuffer(queue, result_flag, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer hiba: %d\n", ret);
        return 1;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_mem);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg (input_mem) hiba: %d\n", ret);
        return 1;
    }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_flag);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg (result_flag) hiba: %d\n", ret);
        return 1;
    }
    ret = clSetKernelArg(kernel, 2, sizeof(int), &array_size);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg (array_size) hiba: %d\n", ret);
        return 1;
    }

    size_t local_size[1] = {1};
    size_t global_size[1] = {NUM_THREADS};

    cl_event event;
    cl_ulong time_start, time_end;
    double total_time;

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, &event);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel hiba: %d\n", ret);
        return 1;
    }

    clFinish(queue);

    ret = clEnqueueReadBuffer(queue, result_flag, CL_TRUE, 0, sizeof(int), &success, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer hiba: %d\n", ret);
        return 1;
    }

    printf("Egy szál sikeresen rendezte a tömböt!\n");

    ret = clEnqueueReadBuffer(queue, input_mem, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer hiba: %d\n", ret);
        return 1;
    }
    printf("Rendezett tömb:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = (double)(time_end - time_start) / 1000000.0;
    printf("Kernel futási ideje: %.3f ms\n", total_time);

    clReleaseMemObject(input_mem);
    clReleaseMemObject(result_flag);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseEvent(event);
    free(source_str);

    return 0;
}