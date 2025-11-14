#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "qmatmul.h"

void ggml_compute_forward_mul_mat_one_chunk(const struct ggml_compute_params * params, struct ggml_tensor * dest, const enum ggml_type compute_type, const int64_t num_rows_per_vec_dot, const int64_t A_row_start, const int64_t A_row_end, const int64_t B_col_start, const int64_t B_col_end){ // calculate tensor C = A * B
    // Get the source tensor out from the destination tensor to be computed
    const struct ggml_tensor * C = dest;
    const struct ggml_tensor * A = C->src[0];
    const struct ggml_tensor * B = C->src[1];

    // Get tensor shape (number of elements per dimension)
    const int A_shape[4] = {A->ne[0], A->ne[1], A->ne[2], A->ne[3]};
    const int B_shape[4] = {B->ne[0], B->ne[1], B->ne[2], B->ne[3]};
    const int C_shape[4] = {C->ne[0], C->ne[1], C->ne[2], C->ne[3]};
    // Get tensor strides in bytes per dimension
    const size_t A_bstride[4] = {A->nb[0], A->nb[1], A->nb[2], A->nb[3]};
    const size_t B_bstride[4] = {B->nb[0], B->nb[1], B->nb[2], B->nb[3]};
    const size_t C_bstride[4] = {C->nb[0], C->nb[1], C->nb[2], C->nb[3]};

    // Some unused parameters
    UNUSED(num_rows_per_vec_dot); // No ARM features
    UNUSED(C_shape);
    UNUSED(B_bstride);

    ggml_vec_dot_t const vec_dot = type_traits_cpu[compute_type].vec_dot; // This is the corresponding vec_dot function's pointer
    enum ggml_type const vec_dot_type = type_traits_cpu[compute_type].vec_dot_type; // The compute type of vec_dot

    // compute broadcast factors of higher dimensions (3 and 2)
    const int broadcast_factor_dim2 = B_shape[2] / A_shape[2];
    const int broadcast_factor_dim3 = B_shape[3] / A_shape[3];

    // If the thread now don't have any work, then just yield
    if (A_row_start >= A_row_end || B_col_start >= B_col_end){
        return;
    }

    // Now find where the right data is stored
    // if B's data type doesn't match the compute vec_dot type, then a convertion would be done by upper functions
    // and the result would be stored in the work buffer of params
    const void * B_data = (B->type == vec_dot_type) ? B->data : params->wdata;
    // And also we are to find the right stride of the right data, this can be calculated by the ggml function below
    const size_t B_data_bstride = ggml_row_size(vec_dot_type, B_shape[0]);

    // Compute by chunking again (block)
    const int BLOCK_SIZE = 16;
    
    // Next, once we calculate 16 (BLOCK_SIZE) results in a loop (in fact it's a col major matrix C's sub-row)
    // First create a buffer to store the result
    float temp[16];

    // Now is the main calculation, iterate by block
    for (int j = B_col_start; j < B_col_end; j += BLOCK_SIZE){
        for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE){
            // In each block, first iterate each col (row in memory) by B
            for (int jj = j; jj < j + BLOCK_SIZE && jj < B_col_end; jj ++){ // jj means a column in B
                // Calculate the corresponding positions of the elements
                int A_indices[4], B_indices[4], C_indices[4]; // indices in 4D Tensor

                B_indices[3] = jj / (B_shape[2] * B_shape[1]);
                B_indices[2] = jj % (B_shape[2] * B_shape[1]) / B_shape[1];
                B_indices[1] = jj % (B_shape[2] * B_shape[1]) % B_shape[1];

                // Then broadcast to the corresponding position in A
                A_indices[3] = B_indices[3] / broadcast_factor_dim3;
                A_indices[2] = B_indices[2] / broadcast_factor_dim2;

                // The position in C is the same as B
                C_indices[1] = B_indices[1];
                C_indices[2] = B_indices[2];
                C_indices[3] = B_indices[3];

                // Use char to calculate the actual position in byte, which is base + offset
                // We use a row in A and a column in B (continuous in memory) to calculate a column in C (continuous in memory)
                const char * A_row = (const char *)A->data + A_indices[2] * A_bstride[2] + A_indices[3] * A_bstride[3];
                const char * B_col = (const char *)B_data + (B_indices[1] + B_indices[2] * B_shape[1] + B_indices[3] * B_shape[2] * B_shape[1]) * B_data_bstride;
                float * C_col = (float *)((char *)C->data + C_indices[1] * C_bstride[1] + C_indices[2] * C_bstride[2] + C_indices[3] * C_bstride[3]);

                // Iterate each row in A to do vec_dot with the same column in B and we'll get a column in C
                for (int ii = i; ii < i + BLOCK_SIZE && ii < A_row_end; ii ++){
                    vec_dot(A_shape[0], temp + (ii - i), 0, A_row + ii * A_bstride[1], 0, B_col, 0, 1);
                }
                
                // Copy the results back to tensor C
                memcpy(C_col + i, temp, (MIN(i + BLOCK_SIZE, A_row_end) - i) * sizeof(float));
            }
        }
    }
}


void rvllm_vec_dot_q4_0_q8_0(int n, float * restrict result, size_t byte_stride_result, const void * restrict vec_x, size_t byte_stride_vec_x, const void * restrict vec_y, size_t byte_stride_vec_y, int num_rows_per_vec_dot){
    // Q8_0 quantizes 32 elements together as a block 
    // Each block has its own scale factor
    const int BLOCK_SIZE = QK8_0;
    const int num_blocks = n / BLOCK_SIZE;

    // Tell the compiler the unused parameters to avoid warnings
    // No impact on the code's performance
    UNUSED(byte_stride_result);
    UNUSED(byte_stride_vec_x);
    UNUSED(byte_stride_vec_y);
    UNUSED(num_rows_per_vec_dot);

    // Recover the original type of vector x and y
    const block_q4_0 * restrict x = vec_x;
    const block_q8_0 * restrict y = vec_y;

    float res = 0.0;
    // First do int mat_mul given the quantization design to get better performance
    for (int block = 0; block < num_blocks; block ++){ // Iterate by BLOCK 
        // For q4_0 quantization, uint8 is used to store two 4-bit elements
        // The element at low 4 bits and the high bits have an offset of BLOCK_SIZE / 2
        // so when doing vec dot, we need to calculate the low-bit result and the high-bit one separately
        int temp_lo = 0; // low-bit temporary result
        int temp_hi = 0; // high-bit temporary result

        for (int i = 0; i < BLOCK_SIZE / 2; i ++){
            // To get the original quantize value from q4_0, we need to:
            // First extract the corresponding 4 bits
            // Then substract offset(8) into [-8, 7]
            const int x_qs_lo = (x[block].qs[i] & 0x0F) - 8; // low q4_0 quantized value: 00001111
            const int x_qs_hi = (x[block].qs[i] >> 4) - 8; // high q4_0 quantized value: 11110000
            // samely we can get the values from vector y
            const int y_qs_lo = y[block].qs[i];
            const int y_qs_hi = y[block].qs[i + BLOCK_SIZE / 2];

            // perform int vec_dot
            temp_lo += x_qs_lo * y_qs_lo;
            temp_hi += x_qs_hi * y_qs_hi;
        }

        int temp = temp_lo + temp_hi; // first merge the result
        
        // Next we need to get the real value by inverse quantization (* scale factor)
        // The real scale factor is stored in a precomputed table, and here d is the index
        res += temp * _GGML_CPU_FP16_TO_FP32(x[block].d) * _GGML_CPU_FP16_TO_FP32(y[block].d);
    }

    // copy the result back
    *result = res;
}