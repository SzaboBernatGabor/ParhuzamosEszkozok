#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include "kernel_loader.h"
#include <time.h>

typedef struct HuffmanNode {
    unsigned char data;
    int frequency;
    struct HuffmanNode *left, *right;
} HuffmanNode;

typedef struct PriorityQueueNode {
    HuffmanNode *node;
    struct PriorityQueueNode *next;
} PriorityQueueNode;

typedef struct PriorityQueue {
    PriorityQueueNode *head;
} PriorityQueue;

PriorityQueue* createPriorityQueue() {
    PriorityQueue *queue = (PriorityQueue*)malloc(sizeof(PriorityQueue));
    queue->head = NULL;
    return queue;
}

void enqueue(PriorityQueue *queue, HuffmanNode *node) {
    PriorityQueueNode *newNode = (PriorityQueueNode*)malloc(sizeof(PriorityQueueNode));
    newNode->node = node;
    newNode->next = NULL;

    if (queue->head == NULL || queue->head->node->frequency > node->frequency) {
        newNode->next = queue->head;
        queue->head = newNode;
    } else {
        PriorityQueueNode *current = queue->head;
        while (current->next != NULL && current->next->node->frequency <= node->frequency) {
            current = current->next;
        }
        newNode->next = current->next;
        current->next = newNode;
    }
}

void generateRandomString(int length, char *output) {
    const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int alphabetSize = sizeof(alphabet) - 1;

    srand((unsigned int)time(NULL));

    for (int i = 0; i < length; i++) {
        output[i] = alphabet[rand() % alphabetSize];
    }

    output[length] = '\0';
}

HuffmanNode* dequeue(PriorityQueue *queue) {
    if (queue->head == NULL) {
        return NULL;
    }
    PriorityQueueNode *temp = queue->head;
    HuffmanNode *node = temp->node;
    queue->head = queue->head->next;
    free(temp);
    return node;
}

HuffmanNode* buildHuffmanTree(int frequencies[]) {
    PriorityQueue *queue = createPriorityQueue();

    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            HuffmanNode *node = (HuffmanNode*)malloc(sizeof(HuffmanNode));
            node->data = (unsigned char)i;
            node->frequency = frequencies[i];
            node->left = node->right = NULL;
            enqueue(queue, node);
        }
    }

    while (queue->head != NULL && queue->head->next != NULL) {
        HuffmanNode *left = dequeue(queue);
        HuffmanNode *right = dequeue(queue);

        HuffmanNode *newNode = (HuffmanNode*)malloc(sizeof(HuffmanNode));
        newNode->data = 0;
        newNode->frequency = left->frequency + right->frequency;
        newNode->left = left;
        newNode->right = right;

        enqueue(queue, newNode);
    }

    HuffmanNode *root = dequeue(queue);
    free(queue);
    return root;
}

void checkError(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Hiba: %s (%d)\n", operation, err);
        exit(1);
    }
}

void generateHuffmanCodes(HuffmanNode *root, int huffmanCodes[], unsigned char codeLengths[], int code, int depth) {
    if (root->left == NULL && root->right == NULL) {
        huffmanCodes[root->data] = code;
        codeLengths[root->data] = depth;
        return;
    }

    if (root->left != NULL) {
        generateHuffmanCodes(root->left, huffmanCodes, codeLengths, (code << 1), depth + 1);
    }

    if (root->right != NULL) {
        generateHuffmanCodes(root->right, huffmanCodes, codeLengths, (code << 1) | 1, depth + 1);
    }
}

int main() {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel calculate_frequencies_kernel, encode_input_kernel;
    cl_int err;

    int error_code;
    char *kernel_source = load_kernel_source("huffman.cl", &error_code);
    if (error_code != 0) {
        fprintf(stderr, "Nem sikerült betölteni a kernel forrást!\n");
        return 1;
    }

    err = clGetPlatformIDs(1, &platform_id, NULL);
    checkError(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    checkError(err, "clGetDeviceIDs");
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    checkError(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(1);
    }

    calculate_frequencies_kernel = clCreateKernel(program, "calculate_frequencies", &err);
    checkError(err, "clCreateKernel (calculate_frequencies)");
    encode_input_kernel = clCreateKernel(program, "encode_input", &err);
    checkError(err, "clCreateKernel (encode_input)");
    
    int frequencies[256] = {0};
    int characters = 2000000;
    char random_string[characters];
    generateRandomString(characters, random_string);

    // Megadható a saját karakterlánc és random generált is

    // const char *input = "AABCBAD";
    const char *input = random_string;


    int input_size = strlen(input);
    printf("Generalt karakterlanc: %s\n", random_string);

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char) * input_size, (void *)input, &err);
    checkError(err, "clCreateBuffer (input_buffer)");
    cl_mem frequencies_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 256, frequencies, &err);
    checkError(err, "clCreateBuffer (frequencies_buffer)");

    err = clSetKernelArg(calculate_frequencies_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(calculate_frequencies_kernel, 1, sizeof(cl_mem), &frequencies_buffer);
    err |= clSetKernelArg(calculate_frequencies_kernel, 2, sizeof(int), &input_size);
    checkError(err, "clSetKernelArg (calculate_frequencies)");

    size_t global_work_size = input_size;
    size_t local_work_size = 256;
    cl_event event1;
    cl_event event2;
    cl_ulong time_start, time_end;
    double total_time;

    err = clEnqueueNDRangeKernel(queue, calculate_frequencies_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event1);
    checkError(err, "clEnqueueNDRangeKernel (calculate_frequencies)");

    err = clEnqueueReadBuffer(queue, frequencies_buffer, CL_TRUE, 0, sizeof(int) * 256, frequencies, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer (frequencies)");

    HuffmanNode *root = buildHuffmanTree(frequencies);

    int huffmanCodes[256] = {0};
    unsigned char codeLengths[256] = {0};
    generateHuffmanCodes(root, huffmanCodes, codeLengths, 0, 0);

    cl_mem huffman_codes_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 256, huffmanCodes, &err);
    checkError(err, "clCreateBuffer (huffman_codes_buffer)");
    cl_mem code_lengths_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * 256, codeLengths, &err);
    checkError(err, "clCreateBuffer (code_lengths_buffer)");
    cl_mem encoded_data_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * input_size, NULL, &err);
    checkError(err, "clCreateBuffer (encoded_data_buffer)");

    err = clSetKernelArg(encode_input_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(encode_input_kernel, 1, sizeof(cl_mem), &huffman_codes_buffer);
    err |= clSetKernelArg(encode_input_kernel, 2, sizeof(cl_mem), &code_lengths_buffer);
    err |= clSetKernelArg(encode_input_kernel, 3, sizeof(cl_mem), &encoded_data_buffer);
    err |= clSetKernelArg(encode_input_kernel, 4, sizeof(int), &input_size);
    checkError(err, "clSetKernelArg (encode_input)");

    err = clEnqueueNDRangeKernel(queue, encode_input_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event2);
    checkError(err, "clEnqueueNDRangeKernel (encode_input)");

    int *encoded_data = (int *)malloc(sizeof(int) * input_size);
    err = clEnqueueReadBuffer(queue, encoded_data_buffer, CL_TRUE, 0, sizeof(int) * input_size, encoded_data, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer (encoded_data)");

    printf("Betuk es frekvenciaik:\n");
    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            printf("'%c': %d\n", i, frequencies[i]);
        }
    }

    printf("Betuk es kodjaik:\n");
    for (int i = 0; i < 256; i++) {
        if (codeLengths[i] > 0) {
            printf("'%c': ", i);
            for (int j = codeLengths[i] - 1; j >= 0; j--) {
                printf("%d", (huffmanCodes[i] >> j) & 1);
            }
            printf("\n");
        }
    }

    // Kikommentelhetjük a mérés miatt
    // printf("Kodolt kimenet (binaris):\n");
    // for (int i = 0; i < input_size; i++) {
    //     int code = encoded_data[i];
    //     int length = codeLengths[(unsigned char)input[i]];
    //     for (int j = length - 1; j >= 0; j--) {
    //         printf("%d", (code >> j) & 1);
    //     }
    // }

    printf("\n");

    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = (double)(time_end - time_start) / 1000000.0;
    printf("Kernel futasi ideje frekvenciak szamolasahoz: %.3f ms\n", total_time);

    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = (double)(time_end - time_start) / 1000000.0;
    printf("Kernel futasi ideje kodolashoz: %.3f ms\n", total_time);

    free(encoded_data);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(frequencies_buffer);
    clReleaseMemObject(huffman_codes_buffer);
    clReleaseMemObject(code_lengths_buffer);
    clReleaseMemObject(encoded_data_buffer);
    clReleaseKernel(calculate_frequencies_kernel);
    clReleaseKernel(encode_input_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_source);

    return 0;
}