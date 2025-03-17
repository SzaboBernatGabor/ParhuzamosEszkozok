__kernel void sample_kernel(__global const float* A, __global const float* B, __global float* C)
{
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

