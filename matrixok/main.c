#include "kernel_loader.h"
#define CL_TARGET_OPENCL_VERSION 220
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

const int MATRIX_SIZE = 10000;

void randomMatrix(float* mat, int size) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

void printMatrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.2f ", mat[i * size + j]);
        }
        printf("\n");
    }
}

double getEventTime(cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (double)(end - start) * 1e-6;  // Nanoseconds to milliseconds
}

int main(void)
{
    int N = MATRIX_SIZE;
    size_t matrixSize = N * N * sizeof(float);

    float *A = (float*)malloc(matrixSize);
    float *B = (float*)malloc(matrixSize);
    float *C = (float*)malloc(matrixSize);
    
    randomMatrix(A, N);
    randomMatrix(B, N);

    // printf("Matrix A:\n");
    // printMatrix(A, N);
    // printf("Matrix B:\n");
    // printMatrix(B, N);

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

    const char *kernel_code = load_kernel_source("matrix.cl", &error_code);
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

    cl_kernel kernel = clCreateKernel(program, "matrix", NULL);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * MATRIX_SIZE, A, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer A. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * MATRIX_SIZE, B, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer B. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * MATRIX_SIZE, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating buffer C. Error code: %d\n", err);
        free(A);
        free(B);
        free(C);
        return 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t local_size = 256;
    size_t groups = (MATRIX_SIZE + local_size - 1) / local_size;
    size_t global_size = groups * local_size;
    cl_event event;

    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
    clFinish(command_queue);
    // clWaitForEvents(1, &event);
    time_t start = clock();
    printf("%d",start);
    printf("Kernel execution finished\n");
    cl_ulong ns;
    printf("Event : %d\n", ns);
    // err = clGetEventProfilingInfo(
    //     event,
    //     CL_PROFILING_COMMAND_QUEUED,
    //     sizeof(ns),
    //     &ns,
    //     0
    // );
    // if (err == CL_PROFILING_INFO_NOT_AVAILABLE) {
    //     printf("Profiling info not available!\n");
    //     return 0;
    // } else if (err != CL_SUCCESS) {
    //     printf("Error code: %d\n", err);
    //     return 0;
    // }
    printf("Queued : %lu\n", ns);
    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(ns),
        &ns,
        NULL
    );
    printf("End : %lu\n", ns);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        d_C,
        CL_TRUE,
        0,
        MATRIX_SIZE * sizeof(int),
        C,
        0,
        NULL,
        NULL
    );
    // cl_event kernelEvent;

    // err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &kernelEvent);
    // checkError(err, "Kernel enqueue");

    // // Eredmények visszaolvasása
    // cl_event readEvent;
    // err = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, matrixSize, C, 1, &kernelEvent, &readEvent);
    // checkError(err, "Buffer read");
    // clWaitForEvents(1, &readEvent);

    // double kernelTime = getEventTime(kernelEvent);
    // double readTime = getEventTime(readEvent);

    // printf("Kernel futási idő: %.3f ms\n", kernelTime);
    // printf("Eredmény visszaolvasási idő: %.3f ms\n", readTime);
    
    err = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, sizeof(float)*MATRIX_SIZE, C, 0, NULL, NULL);

    printf("lefutott");

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}