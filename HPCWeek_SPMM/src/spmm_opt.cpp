#include <omp.h>
#include <cstring>
#include <cstddef>
#include <algorithm>
#include <arm_sve.h>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// 下述 tile 可按机器 L1/L2 调整：L1 友好行块；L2 友好 k/n 块
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void spmm_cpu_opt(
    const int* __restrict__ ptr,   // CSR row ptr, length num_v+1
    const int* __restrict__ idx,   // CSR col idx, length nnz
    const float* __restrict__ val, // CSR values, length nnz
    const float* __restrict__ vin, // Dense B (K x INFEATURE), row-major
    float* __restrict__ vout,      // Output C (num_v x INFEATURE), row-major
    const int num_v,
    const int INFEATURE,
    int _k)
{
    if(INFEATURE ==2048) {

        #pragma omp parallel for num_threads(32) schedule(dynamic, 4)
        for (int m = 0; m < num_v; ++m) {
            const int begin = ptr[m];
            const int end   = ptr[m+1];
    
            float* __restrict__ out_row = vout + (size_t)m * INFEATURE;
    
            // 矢量化清零整行
            {
                int j = 0;
                // 也可用 memset，但 SVE 清零利于保持矢量化一致
                for (; j < INFEATURE; ) {
                    svbool_t pg = svwhilelt_b32(j, INFEATURE);
                    svst1_f32(pg, out_row + j, svdup_f32(0.0f));
                    j += svcntw(); // 每次前进 #float32_per_vector
                }
            }
    
            // 对该行的每个非零，做一次 AXPY： out_row += a * vin[row_j, :]
            const int nnz_row = end - begin;
            #pragma GCC ivdep
            for (int i = begin; i < end; ++i) {
                const int   col = idx[i];
                const float a   = val[i];
    
                const float* __restrict__ b_row = vin + (size_t)col * INFEATURE;
    
                // 预取下一非零的 b_row，减少流水停顿
                // if (i + 1 < end) {
                //     const float* next_b = vin + (size_t)idx[i + 1] * INFEATURE;
                //     __builtin_prefetch(next_b, 0, 3);  // 读取，高局部性
                // }
    
                const svfloat32_t a_vec = svdup_f32(a);
    
                // 4路展开，利用更多 SVE 寄存器和指令级并行
                int j = 0;
                const int vec_len = svcntw();
                const int unroll = 4;
                const int block_size = vec_len * unroll;
                
                // 主循环：4路展开，重组指令以提高流水线效率
                for (; j + block_size <= INFEATURE; j += block_size) {
                    svbool_t pg = svptrue_b32();
                    
                    // 批量加载 - 减少 load-use 延迟
                    svfloat32_t c0 = svld1_f32(pg, out_row + j);
                    svfloat32_t c1 = svld1_f32(pg, out_row + j + vec_len);
                    svfloat32_t c2 = svld1_f32(pg, out_row + j + vec_len*2);
                    svfloat32_t c3 = svld1_f32(pg, out_row + j + vec_len*3);
                    
                    svfloat32_t b0 = svld1_f32(pg, b_row + j);
                    svfloat32_t b1 = svld1_f32(pg, b_row + j + vec_len);
                    svfloat32_t b2 = svld1_f32(pg, b_row + j + vec_len*2);
                    svfloat32_t b3 = svld1_f32(pg, b_row + j + vec_len*3);
                    
                    // FMA 操作 - 充分利用流水线
                    c0 = svmla_f32_z(pg, c0, b0, a_vec);
                    c1 = svmla_f32_z(pg, c1, b1, a_vec);
                    c2 = svmla_f32_z(pg, c2, b2, a_vec);
                    c3 = svmla_f32_z(pg, c3, b3, a_vec);
                    
                    // 批量存储
                    svst1_f32(pg, out_row + j, c0);
                    svst1_f32(pg, out_row + j + vec_len, c1);
                    svst1_f32(pg, out_row + j + vec_len*2, c2);
                    svst1_f32(pg, out_row + j + vec_len*3, c3);
                }
                
                // 尾部处理
                for (; j < INFEATURE; ) {
                    svbool_t pg = svwhilelt_b32(j, INFEATURE);
                    svfloat32_t c = svld1_f32(pg, out_row + j);
                    svfloat32_t b = svld1_f32(pg, b_row   + j);
                    c = svmla_f32_m(pg, c, b, a_vec);
                    svst1_f32(pg, out_row + j, c);
                    j += vec_len;
                }
            }
        }
        return;
    }
    // ---------- 维度与 tile 设定 ----------
    const int nnz = ptr[num_v];

    int K = _k;

    // 基于 L1/L2 的经验值，可按机器调整
    const int tile_m = 64;   // L1: 一次处理的 A 的行块
    int tile_k = 512;  // L2: 归约维分块
    const int tile_n = 128;  // L2: B/C 的列块

    //按照固定的输入形状选择算法
    if((INFEATURE == 10240 && num_v==785)) {
        tile_k = 384;
    }else if((INFEATURE == 20480 &&num_v==2529)||(INFEATURE == 4096 &&num_v==3557)||(INFEATURE == 10240 && num_v==10240)) {
        tile_k = 768;
    }

    const int Tm = ceil_div(num_v,    tile_m);
    const int Tk = ceil_div(K,        tile_k);
    const int Tn = ceil_div(INFEATURE, tile_n);

    // ---------- 预计算：每行 × 每 k-tile 的 CSR 区间 ----------
    // block_starts/ends 大小：num_v * Tk
    int *block_starts = (int*)malloc(sizeof(int) * num_v * Tk);
    int *block_ends   = (int*)malloc(sizeof(int) * num_v * Tk);

    // 二分 + 小范围线性，和你 tiling_opt 的写法一致
    for (int r = 0; r < num_v; ++r) {
        const int rbeg = ptr[r], rend = ptr[r+1];

        for (int k_idx = 0; k_idx < Tk; ++k_idx) {
            const int k0 = k_idx * tile_k;
            const int k1 = std::min(k0 + tile_k, K);

            // lower_bound(idx[rbeg:rend), k0]
            int low = rbeg, high = rend;
            while (high - low > 32) {
                int mid = low + ((high - low) >> 1);
                if (idx[mid] < k0) low = mid + 1; else high = mid;
            }
            while (low < high && idx[low] < k0) ++low;
            block_starts[r*Tk + k_idx] = low;

            // lower_bound(idx[rbeg:rend), k1]
            int low2 = low, high2 = rend;
            while (high2 - low2 > 32) {
                int mid = low2 + ((high2 - low2) >> 1);
                if (idx[mid] < k1) low2 = mid + 1; else high2 = mid;
            }
            while (low2 < high2 && idx[low2] < k1) ++low2;
            block_ends[r*Tk + k_idx] = low2;
        }
    }

    // ---------- 并行：按 n-tile 分工（天然无写冲突） ----------
    #pragma omp parallel
    {
        // 为每个 j-tile 预先分配所有 k-tile 的 B_tiles 数组
        float **B_tiles = (float**)malloc(sizeof(float*) * Tk);
        for (int k = 0; k < Tk; ++k) {
            B_tiles[k] = (float*)malloc(sizeof(float) * tile_k * tile_n);
        }

        const int vl = svcntw(); // 每个向量包含的 float32 数

        #pragma omp for schedule(dynamic) 
        for (int j_blk = 0; j_blk < Tn; ++j_blk) {
            const int colB_start = j_blk * tile_n;
            const int colB_end   = std::min(colB_start + tile_n, INFEATURE);
            const int colB_len   = colB_end - colB_start;

            // ---------- 预先打包所有 k-tile 的 B 数据 ----------
            for (int k_blk = 0; k_blk < Tk; ++k_blk) {
                const int k0 = k_blk * tile_k;
                const int k1 = std::min(k0 + tile_k, K);
                const int cur_k_len = k1 - k0;
                
                float *B_tile = B_tiles[k_blk];

                // pack B(k0:k1, colB_start:colB_end) 到 B_tile（行主序）
                for (int bk = 0; bk < cur_k_len; ++bk) {
                    const float * __restrict__ src = vin + (size_t)(k0 + bk) * INFEATURE + colB_start;
                    float * __restrict__ dst = B_tile + (size_t)bk * colB_len;

                    // SVE 搬运
                    int j = 0;
                    for (; j + vl <= colB_len; j += vl) {
                        svbool_t pg = svptrue_b32();
                        svst1_f32(pg, dst + j, svld1_f32(pg, src + j));
                    }
                    if (j < colB_len) {
                        svbool_t pg = svwhilelt_b32(j, colB_len);
                        svst1_f32(pg, dst + j, svld1_f32(pg, src + j));
                    }
                }
            }

            // ---------- 按行块和行处理，C-寄存器驻留优化 ----------
            for (int i_blk = 0; i_blk < Tm; ++i_blk) {
                const int rowA_start = i_blk * tile_m;
                const int rowA_end   = std::min(rowA_start + tile_m, num_v);

                for (int ii = rowA_start; ii < rowA_end; ++ii) {
                    float * __restrict__ C_row = vout + (size_t)ii * INFEATURE + colB_start;

                    // 按 j-chunk (4*VL) 分块处理，每个 chunk 做寄存器驻留
                    const int unroll = 4;
                    const int step = vl * unroll;
                    
                    int j = 0;
                    // 处理完整的 4*VL 块
                    for (; j + step <= colB_len; j += step) {
                        svbool_t pg = svptrue_b32();
                        
                        // 初始化 C 寄存器为 0（寄存器驻留开始）
                        svfloat32_t c0 = svdup_f32(0.0f);
                        svfloat32_t c1 = svdup_f32(0.0f);
                        svfloat32_t c2 = svdup_f32(0.0f);
                        svfloat32_t c3 = svdup_f32(0.0f);

                        // 遍历所有 k-tile，累加到寄存器
                        for (int k_blk = 0; k_blk < Tk; ++k_blk) {
                            const int k0 = k_blk * tile_k;
                            const int kk_begin = block_starts[ii*Tk + k_blk];
                            const int kk_end   = block_ends  [ii*Tk + k_blk];
                            
                            float *B_tile = B_tiles[k_blk];

                            // 对该行在当前 k-tile 的所有非零元素
                            for (int p = kk_begin; p < kk_end; ++p) {
                                const int   col = idx[p];
                                const float a   = val[p];
                                const int   bro = col - k0;
                                const float * __restrict__ B_row = B_tile + (size_t)bro * colB_len;

                                // 预取
                                if (likely(p + 1 < kk_end)) {
                                    const int next_bro = idx[p + 1] - k0;
                                    __builtin_prefetch(B_tile + (size_t)next_bro * colB_len + j, 0, 3);
                                }

                                const svfloat32_t a_vec = svdup_f32(a);

                                // 从 B_tile 加载并累加到 C 寄存器
                                svfloat32_t b0 = svld1_f32(pg, B_row + j);
                                svfloat32_t b1 = svld1_f32(pg, B_row + j + vl);
                                svfloat32_t b2 = svld1_f32(pg, B_row + j + 2*vl);
                                svfloat32_t b3 = svld1_f32(pg, B_row + j + 3*vl);

                                c0 = svmla_f32_x(pg, c0, b0, a_vec);
                                c1 = svmla_f32_x(pg, c1, b1, a_vec);
                                c2 = svmla_f32_x(pg, c2, b2, a_vec);
                                c3 = svmla_f32_x(pg, c3, b3, a_vec);
                            }
                        } // end k_blk

                        // 一次性写回 C（寄存器驻留结束）
                        svst1_f32(pg, C_row + j,           c0);
                        svst1_f32(pg, C_row + j + vl,      c1);
                        svst1_f32(pg, C_row + j + 2*vl,    c2);
                        svst1_f32(pg, C_row + j + 3*vl,    c3);
                    } // end j (full chunks)

                    // 处理尾部（逐 VL 块）
                    while (j < colB_len) {
                        svbool_t pg = svwhilelt_b32(j, colB_len);
                        svfloat32_t c = svdup_f32(0.0f);

                        // 遍历所有 k-tile
                        for (int k_blk = 0; k_blk < Tk; ++k_blk) {
                            const int k0 = k_blk * tile_k;
                            const int kk_begin = block_starts[ii*Tk + k_blk];
                            const int kk_end   = block_ends  [ii*Tk + k_blk];
                            
                            float *B_tile = B_tiles[k_blk];

                            for (int p = kk_begin; p < kk_end; ++p) {
                                const int   col = idx[p];
                                const float a   = val[p];
                                const int   bro = col - k0;
                                const float * __restrict__ B_row = B_tile + (size_t)bro * colB_len;

                                const svfloat32_t a_vec = svdup_f32(a);
                                svfloat32_t b = svld1_f32(pg, B_row + j);
                                c = svmla_f32_m(pg, c, b, a_vec);
                            }
                        }

                        // 写回尾部
                        svst1_f32(pg, C_row + j, c);
                        j += vl;
                    } // end tail
                } // end ii
            } // end i_blk
        } // end j_blk

        // 释放所有 B_tiles
        for (int k = 0; k < Tk; ++k) {
            free(B_tiles[k]);
        }
        free(B_tiles);
    } // end parallel

    free(block_starts);
    free(block_ends);
}
