__kernel void calculate_frequencies(__global const uchar *input,
                                    __global int *frequencies,
                                    int input_size) {
    int gid = get_global_id(0);

    if (gid < input_size) {
        atomic_inc(&frequencies[input[gid]]);
    }
}

__kernel void encode_input(__global const uchar *input,
                           __global int *huffman_codes,
                           __global uchar *code_lengths,
                           __global int *encoded_data,
                           int input_size) {
    int gid = get_global_id(0);

    if (gid < input_size) {
        uchar symbol = input[gid];
        encoded_data[gid] = huffman_codes[symbol];
    }
}