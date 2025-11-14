#pragma GCC optimize("inline-functions,unroll-loops")
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qmatmul.h"
#include <omp.h>

# define likely(x)	__builtin_expect(!!(x), 1)
# define unlikely(x)	__builtin_expect(!!(x), 0)

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

// 用于去重输出的参数配置结构
typedef struct {
    int B_col_start;
    int B_col_end;
    // int A_row_start;
    // int A_row_end;
    int B_shape[2];
} ParamConfig;

// 简单的哈希表实现（固定大小）
#define MAX_CONFIGS 1024*64
static ParamConfig seen_configs[MAX_CONFIGS];
static int num_configs = 0;

// 检查配置是否已存在，不存在则添加并返回1（新配置），存在返回0
static int check_and_add_config(int B_col_start, int B_col_end, int B_shape_1, int B_shape_2) {
    // 线性搜索检查是否已存在
    for (int i = 0; i < num_configs; i++) {
        if (seen_configs[i].B_col_start == B_col_start &&
            seen_configs[i].B_col_end == B_col_end &&
            // seen_configs[i].A_row_start == A_row_start &&
            // seen_configs[i].A_row_end == A_row_end &&
            seen_configs[i].B_shape[1] == B_shape_1 &&
            seen_configs[i].B_shape[2] == B_shape_2) {
            return 0; // 已存在
        }
    }
    
    // 不存在，添加新配置
    if (num_configs < MAX_CONFIGS) {
        seen_configs[num_configs].B_col_start = B_col_start;
        seen_configs[num_configs].B_col_end = B_col_end;
        // seen_configs[num_configs].A_row_start = A_row_start;
        // seen_configs[num_configs].A_row_end = A_row_end;
        seen_configs[num_configs].B_shape[1] = B_shape_1;
        seen_configs[num_configs].B_shape[2] = B_shape_2;
        num_configs++;
        return 1; // 新配置
    }
    
    return 0; // 数组已满
}

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

    // ==== 优化路径 1: B_shape[2]=1 (占 >95% 的情况) ====
    // 所有日志数据显示 B_shape[2] 都是 1，无需第3维广播计算
    if (likely(B_shape[2] == 1)) {
        const char * A_row_base = (const char *)A->data;  // A_indices[2]=0, A_indices[3]=0
        
        // 子优化：B_shape[1]=20 且列数=10 (最常见，占 >80%)
        if (likely(B_shape[1] == 20 && (B_col_end - B_col_start) == 10)) {
            // 完全展开循环处理 10 列
            const int64_t num_cols = B_col_end - B_col_start;
            
            for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE) {
                const int block_rows = MIN(BLOCK_SIZE, A_row_end - i);
                
                // 展开处理 10 列
                for (int64_t jj = B_col_start; jj < B_col_end; jj++) {
                    const char * B_col = (const char *)B_data + jj * B_data_bstride;
                    float * C_col = (float *)((char *)C->data + jj * C_bstride[1]);
                    
                    // 向量化处理行块
                    for (int ii = 0; ii < block_rows; ii++) {
                        vec_dot(A_shape[0], temp + ii, 0, 
                               A_row_base + (i + ii) * A_bstride[1], 0, 
                               B_col, 0, 1);
                    }
                    
                    // 批量写回结果
                    memcpy(C_col + i, temp, block_rows * sizeof(float));
                }
            }
        }
        // 子优化：B_shape[1]=1 (单列)
        else if (likely(B_shape[1] == 1)) {
            const char * B_col = (const char *)B_data;
            float * C_col = (float *)C->data;
            
            // 一次性处理整个列
            for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE) {
                const int block_rows = MIN(BLOCK_SIZE, A_row_end - i);
                
                for (int ii = 0; ii < block_rows; ii++) {
                    vec_dot(A_shape[0], temp + ii, 0,
                           A_row_base + (i + ii) * A_bstride[1], 0,
                           B_col, 0, 1);
                }
                
                memcpy(C_col + i, temp, block_rows * sizeof(float));
            }
        }
        // 子优化：B_shape[1]=2 (双列)
        else if (likely(B_shape[1] == 2)) {
            for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE) {
                const int block_rows = MIN(BLOCK_SIZE, A_row_end - i);
                
                // 手动展开 2 列
                for (int64_t jj = 0; jj < 2; jj++) {
                    const char * B_col = (const char *)B_data + jj * B_data_bstride;
                    float * C_col = (float *)((char *)C->data + jj * C_bstride[1]);
                    
                    for (int ii = 0; ii < block_rows; ii++) {
                        vec_dot(A_shape[0], temp + ii, 0,
                               A_row_base + (i + ii) * A_bstride[1], 0,
                               B_col, 0, 1);
                    }
                    
                    memcpy(C_col + i, temp, block_rows * sizeof(float));
                }
            }
        }
        // 通用 B_shape[2]=1 路径
        else {
            for (int64_t j = B_col_start; j < B_col_end; j++) {
                const char * B_col = (const char *)B_data + j * B_data_bstride;
                float * C_col = (float *)((char *)C->data + j * C_bstride[1]);
                
                for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE) {
                    const int block_rows = MIN(BLOCK_SIZE, A_row_end - i);
                    
                    for (int ii = 0; ii < block_rows; ii++) {
                        vec_dot(A_shape[0], temp + ii, 0,
                               A_row_base + (i + ii) * A_bstride[1], 0,
                               B_col, 0, 1);
                    }
                    
                    memcpy(C_col + i, temp, block_rows * sizeof(float));
                }
            }
        }
    } 
    // ==== 极少见：通用路径 (B_shape[2]>1) ====
    else {
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

    // ============ RISC-V Vector Extension Optimized Version ============
    float res = 0.0f;
    
    // 处理每个 block (每个 block 有 32 个元素)
    // #pragma omp parallel for reduction(+:res) num_threads(2) schedule(static)
    for (int block = 0; block < num_blocks; block++) {
        // 设置向量长度为 16 (处理 BLOCK_SIZE/2 个 uint8_t，每个包含2个4位元素)
        size_t vl = __riscv_vsetvl_e8m1(16);
        
        // 加载 Q4_0 数据 (16 个 uint8_t，包含 32 个 4-bit 元素)
        vuint8m1_t vx_packed = __riscv_vle8_v_u8m1(x[block].qs, vl);
        
        // 加载 Q8_0 数据的低 16 个元素
        vint8m1_t vy_lo = __riscv_vle8_v_i8m1(&y[block].qs[0], vl);
        // 加载 Q8_0 数据的高 16 个元素
        vint8m1_t vy_hi = __riscv_vle8_v_i8m1(&y[block].qs[16], vl);
        
        // 解包 Q4_0: 提取低 4 位
        vuint8m1_t vx_lo_u = __riscv_vand_vx_u8m1(vx_packed, 0x0F, vl);
        // 转换为有符号并减去偏移 8
        vint8m1_t vx_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vx_lo_u), 8, vl);
        
        // 解包 Q4_0: 提取高 4 位
        vuint8m1_t vx_hi_u = __riscv_vsrl_vx_u8m1(vx_packed, 4, vl);
        // 转换为有符号并减去偏移 8
        vint8m1_t vx_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vx_hi_u), 8, vl);
        
        // 执行向量乘法: vx_lo * vy_lo 和 vx_hi * vy_hi
        vint16m2_t vprod_lo = __riscv_vwmul_vv_i16m2(vx_lo, vy_lo, vl);
        vint16m2_t vprod_hi = __riscv_vwmul_vv_i16m2(vx_hi, vy_hi, vl);
        
        // 合并低位和高位的乘积
        vint16m2_t vprod_sum = __riscv_vadd_vv_i16m2(vprod_lo, vprod_hi, vl);
        
        // 扩展到 int32 并归约求和
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vsum = __riscv_vwredsum_vs_i16m2_i32m1(vprod_sum, vzero, vl);
        
        // 提取标量求和结果
        int32_t block_sum = __riscv_vmv_x_s_i32m1_i32(vsum);
        
        // 乘以 scale 因子
        float scale_x = _GGML_CPU_FP16_TO_FP32(x[block].d);
        float scale_y = _GGML_CPU_FP16_TO_FP32(y[block].d);
        res += block_sum * scale_x * scale_y;
    }
    
    *result = res;

}