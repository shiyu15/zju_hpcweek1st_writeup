#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"
#include "vec.h"

void rvllm_vec_dot_q4_0_q8_0(int n, float * restrict result, size_t byte_stride_result, 
                             const void * restrict vec_x, size_t byte_stride_vec_x, 
                             const void * restrict vec_y, size_t byte_stride_vec_y, 
                             int num_rows_per_vec_dot);

const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = {
        .from_float = quantize_row_q4_0,
        .vec_dot = rvllm_vec_dot_q4_0_q8_0,
        .vec_dot_type = GGML_TYPE_Q8_0,
        .nrows = 1
    },
    [GGML_TYPE_Q8_0] = {
        .from_float = quantize_row_q8_0,
        .vec_dot = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type = GGML_TYPE_Q8_0,
        .nrows = 1
    },
    [GGML_TYPE_Q6_K] = {
        .from_float = quantize_row_q6_K,
        .vec_dot = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type = GGML_TYPE_Q8_K,
        .nrows = 1
    },
    [GGML_TYPE_Q8_K] = {
        .from_float = quantize_row_q8_K,
    }
};