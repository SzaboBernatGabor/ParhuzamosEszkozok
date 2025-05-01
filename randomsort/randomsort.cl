uint rand_custom(uint* seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return *seed;
}

__kernel void random_sort(__global int* input, __global atomic_int* success_flag, const int size) {
    int id = get_global_id(0);

    uint seed = (uint)(id + 1) * 123456789;

    int local_data[64];
    if (size > 64) return;

    for (int i = 0; i < size; i++) {
        local_data[i] = input[i];
    }

    while (atomic_load(success_flag) == 0) {
        for (int i = size - 1; i > 0; i--) {
            int j = rand_custom(&seed) % (i + 1);
            int temp = local_data[i];
            local_data[i] = local_data[j];
            local_data[j] = temp;
        }

        int sorted = 1;
        for (int i = 1; i < size; i++) {
            if (local_data[i-1] > local_data[i]) {
                sorted = 0;
                break;
            }
        }

        if (sorted) {
            atomic_store(success_flag, 1);
            for (int i = 0; i < size; i++) {
                input[i] = local_data[i];
            }
            return;
        }
    }
}