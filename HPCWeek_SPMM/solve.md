# spmm优化记录
## 1.参考实现
先来看一开始的参考实现
```cpp
void spmm_cpu_ref(const int* __restrict__ ptr, const int* __restrict__ idx, const float* __restrict__ val, const float* __restrict__ vin, float* __restrict__ vout, const int num_v, const int INFEATURE, const int k)
{
// 遍历每一行
#pragma omp parallel for schedule(static)
    for (int m = 0; m < num_v; ++m) {
        int begin = ptr[m], end = ptr[m + 1];
        // 遍历每个特征维度
        for (int j = 0; j < INFEATURE; ++j) {
            float result = 0.0f;
            // 计算稀疏矩阵第m行与输入矩阵第j列的点积
            for (int i = begin; i < end; ++i) {
                result += vin[idx[i] * INFEATURE + j] * val[i];
            }
            vout[m * INFEATURE + j] = result;
        }
    }
}

```
参考实现的方法是对稀疏矩阵按行做划分，每一个线程处理相邻的若干行。对稀疏矩阵A每一行的每一个元素`val[i]`来说，就得和稠密矩B阵的对应的列的每一个元素`vin[idx[i] * INFEATURE + j]`去做乘法。这种矩阵乘法方式和我们大一线性代数学习的矩阵乘法方式一样，让矩阵A的一行，和矩阵B的一列做点积，再写到结果矩阵的一个元素里。这样的好处就是可以把点积的中间值放在寄存器里，读写速度比较快，最后只需要往结果里写一次。坏处是矩阵B是按行存储的，按列取B矩阵的速度比较慢。
这样有两个问题，一个是`static`这种固定的分块方式虽然会减少调度开销，但是稀疏矩阵A每行的元素数量不固定，因此可能几个线程要处理的元素的数量不太一样，导致负载不太均匀。快的线程需要等待慢的线程计算。
第二是下面这个最内部循环的空间局部性很差，`i`每次迭代要访问的`vin[idx[i] * INFEATURE + j]`在内存中不是相邻的，需要反复地重新取cache。
```CPP
for (int i = begin; i < end; ++i) {
    result += vin[idx[i] * INFEATURE + j] * val[i];
}
```

## 2.简单调整访存顺序

针对上面的写法，我们可以有下面的优化代码。这个代码的原始性能分差不多能到740左右。
```cpp
#include <omp.h>
#include <cstring>
#include <cstddef>
#include "spmm_opt.h"
#include <arm_sve.h>

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
    // OpenMP：按行并行
    #pragma omp parallel for schedule(dynamic,4)
    for (int m = 0; m < num_v; ++m) {
        const int begin = ptr[m];
        const int end   = ptr[m+1];

        float* __restrict__ out_row = vout + (size_t)m * INFEATURE;

        // 矢量化清零整行
        {
            int j = 0;
            for (; j < INFEATURE; ) {
                svbool_t pg = svwhilelt_b32(j, INFEATURE);
                svst1_f32(pg, out_row + j, svdup_f32(0.0f));
                j += svcntw();
            }
        }

        // 对该行的每个非零，做一次 AXPY： out_row += a * vin[row_j, :]
        for (int i = begin; i < end; ++i) {
            const int   col = idx[i];
            const float a   = val[i];

            const float* __restrict__ b_row = vin + (size_t)col * INFEATURE;
            const svfloat32_t a_vec = svdup_f32(a);

            int j = 0;
            for (; j < INFEATURE; ) {
                svbool_t pg = svwhilelt_b32(j, INFEATURE);
                svfloat32_t c = svld1_f32(pg, out_row + j);
                svfloat32_t b = svld1_f32(pg, b_row   + j);

                // c = c + a*b
                c = svmla_f32_m(pg, c, b, a_vec);

                svst1_f32(pg, out_row + j, c);
                j += svcntw();
            }
        }
    }
}
```
这次改变了矩阵乘法的计算方法，改为让矩阵A中的每个元素`val[i]`和B的对应行做乘法，然后将结果加到矩阵C的结果行中。这样一来，对矩阵B和矩阵C的内存访问都是连续的，提高了访存的效率。而且不会有对C的写冲突。

同时将openMP的调度改为`#pragma omp parallel for schedule(dynamic,4)`就是按照稀疏矩阵A的4行为单位进行多线程调度。哪个线程结束了之前4行的计算任务，就再去取新的未计算的四行进行计算。这样可以让几个线程上的负载尽可能均衡。dynamic的参数换成1或者8也可以，但是效果可能对某些矩阵比较好，对某些矩阵比较差。

还使用了arm上的sve指令，例如fma操作，可以在1个周期内做乘加两个操作。

之后又在循环里面做一些循环展开的优化，但是效果也不是很明显，能再涨几分。

现在的程序的主要问题有两个：1.是让稀疏矩阵A中的每个元素`val[i]`和稠密矩阵B的对应行相乘的时候，因为B的行太长了，cache放不下，下次需要乘B的相同行（例如第k行）的时候无法复用这些数据。而且稀疏矩阵下次出现第k列也不知道是什么时候，更无法去复用B的这一行了。2.结果矩阵C的中间结果需要多次写到内存中，而不是像参考的矩阵乘法中只需要写一次，增加了读写开销。

## 3. 进行tiling

因此，我们又有了下面tiling的实现。平台上性能分差不多是2000分左右。

在这个实现里，对稀疏矩阵A按照行m来分块、对稠密矩阵B按照k和n进行了分块。每个线程处理一部分稠密矩阵B的列。这样也不会有写冲突。
```cpp
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
    // ---------- 维度与 tile 设定 ----------
    const int nnz = ptr[num_v];

    int K = _k;

    const int tile_m = 64;  
    const int tile_k = 512;  
    const int tile_n = 128;  

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

```

下面我们一步步来看上面的代码。

先进行tiling分块，按下面这个单位来进行分块。为了表述方便我们令矩阵A的形状为`M*K`,矩阵B的形状为`K*N`。这个参数对不同的测试用例效果不太一样，因为测试用例的形状已知而且固定，所以之后可以改变这个参数来做针对性的优化。
```cpp
const int tile_m = 64;  
const int tile_k = 512;  
const int tile_n = 128;  
```

之后，在代码的这个部分，我们先按照k的分块，去标明稀疏矩阵A每行的k方向上分块的起点和终点。
```cpp
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
```

比如说只看A中的某一行r，是这样分块的。
```text
idx:   [  3   7  20  530  531  900 1200  ... ]
         |   |   |    |    |    |    |
         v   v   v    v    v    v    v
K 分块: [0........512)[512......1024)[1024...]

对这行 r:
k_idx=0: 非零是 idx[0..2]
k_idx=1: 非零是 idx[3..5]
k_idx=2: 非零是 idx[6...]

因此有:
block_starts[r,0]=0 block_ends[r,0] = 3
block_starts[r,1]=3 block_ends[r,1] = 6
block_starts[r,2]=6

```
这样操作下来，方便之后遍历A的每行，可以快速地找到某个k分块。能够减少之后计算负载中的条件判断和分支。


之后按照n的分块来进行并行。
输出矩阵C的列被切成`Tn`个`tile_n=128`宽的竖条，每个 `j_blk` 一竖条。每一个竖条由一个线程处理。

```
C (M x N)

  [         ] [         ] [         ] ...
  [         ] [         ] [         ] ...
  [ j_blk=0 ] [ j_blk=1 ] [ j_blk=2 ] ...
  [         ] [         ] [         ] ...
```
要得到C的这么一个竖条，每个线程也需要B的一个竖条去计算，为了让B的一个竖条在内存中连续，我们进行下面的tiling。将这个线程要用到的B的竖条存入这个线程自己的B_tiling缓存中。这样会让更换B中列号时，从增加几个B矩阵的width变成只增加几个tile_k=512。对cache更加友好。
```cpp
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
    }
}

```

最后就是进行实际的计算了。固定B中的一个tile，让这个小tile和不同的A中的元素相乘。然后将计算结果存到寄存器中。最后再一次写入C中的对应位置。同时也使用了四路的展开来加速。
```cpp
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
    }
}
```


## 4.进一步优化
这样做下来性能已经很不错了，但是还能再继续做简单的优化。

tiling之后的版本对第三个测试的性能不如tiling之前的版本，可能是形状的问题？因此可以通过检测传入的矩阵的形状来选择使用哪种算法。

我们还可以进行调参，看看测试数据在什么参数上性能好一些。
最后测试下来，我这个性能好像就不错。这个可能和cpu的cache大小相关。但升腾的cpu很难获取cache大小的数据，比较难解释原因。
```cpp
    const int tile_m = 64;  
    int tile_k = 512;  
    const int tile_n = 128;  

    //按照固定的输入形状选择算法
    if((INFEATURE == 10240 && num_v==785)) {
        tile_k = 384;
    }else if((INFEATURE == 20480 &&num_v==2529)||(INFEATURE == 4096 &&num_v==3557)||(INFEATURE == 10240 && num_v==10240)) {
        tile_k = 768;
    }
```
这样下来性能分差不多在2400分左右。

## 5.课外阅读
现今关于spmm的论文绝大多数是gpu上的，很少有cpu上的，我看了几篇

这篇开源的论文说arm上可以进行分块的优化
Optimizing massively parallel sparse matrix computing on ARM many-core
processor
但是这篇论文的开源代码有问题，作者没有最关键的实现，反而写了很多没用过的工具类，无法成功运行。

这篇是讲如何用即时编译优化spmm的，但可惜针对的是width很小的密集矩阵。这样密集矩阵的一行就可以放进x86的几个avx512寄存器内。
JITSPMM: Just-in-Time Instruction Generation for Accelerated Sparse Matrix-Matrix Multiplication

## 附录
最终提交代码
```cpp
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

```

