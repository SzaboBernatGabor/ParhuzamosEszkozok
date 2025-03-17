#include "kernel_loader.h"
#define CL_TARGET_OPENCL_VERSION 220
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const int SAMPLE_SIZE = 20000000;

int main(void)
{
    float *A = (float *)malloc(SAMPLE_SIZE * sizeof(float));
    float *B = (float *)malloc(SAMPLE_SIZE * sizeof(float));
    float *C = (float *)malloc(SAMPLE_SIZE * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        printf("[ERROR] Memory allocation failed\n");
        return 0;
    }

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        A[i] = i;
        B[i] = i + 1;
    }

    int i;
    cl_int err;
    int error_code;

    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    const char *kernel_code = load_kernel_source("sample.cl", &error_code);
    if (error_code != 0)
    {
        printf("Source code loading error!\n");
        free(A);
        free(B);
        free(C);
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    const char options[] = "";
    err = clBuildProgram(
        program,
        1,
        &device_id,
        options,
        NULL,
        NULL);
    if (err != CL_SUCCESS)
    {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size);
        char *build_log = (char *)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size);
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        free(A);
        free(B);
        free(C);
        return 0;
    }
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    cl_kernel kernel = clCreateKernel(program, "sample_kernel", NULL);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * SAMPLE_SIZE, A, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer A. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * SAMPLE_SIZE, B, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer B. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * SAMPLE_SIZE, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer C. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);

    size_t global_size = SAMPLE_SIZE;
    size_t local_size = 64;
    cl_event event;

    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
    clFinish(command_queue);
    err = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeof(float)*SAMPLE_SIZE, C, 0, NULL, NULL);

    printf("lefutott");

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}