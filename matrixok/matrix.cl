#define TILE_SIZE 2

__kernel void matrix(__global float* A, __global float* B, __global float* C, int N) {
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    
    float sum = 0.0f;

    for (int i = 0; i < N / TILE_SIZE; i++) {
        Asub[localRow][localCol] = A[row * N + (i * TILE_SIZE + localCol)];
        Bsub[localRow][localCol] = B[(i * TILE_SIZE + localRow) * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[row * N + col] = sum;
}
