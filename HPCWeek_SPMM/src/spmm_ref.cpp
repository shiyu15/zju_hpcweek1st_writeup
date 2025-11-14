#include "spmm_ref.h"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>

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
