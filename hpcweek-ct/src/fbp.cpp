#pragma GCC optimize("Ofast,fast-math,inline-functions,unroll-loops")

#include <cmath>
#include <algorithm>
#include <vector>
#include <bits/stdc++.h>
#include <omp.h>
#include <arm_neon.h>

#include "fbp.h"

constexpr double PI = 3.14159265358979323846;

/**
 * Generate Ramp filter kernel in spatial domain
 * 
 * @param len  Kernel length (will be made odd if even)
 * @param d    Detector pixel spacing (default 1.0)
 * @return     Symmetric filter kernel centered at middle
 */
static std::vector<float> ramp_kernel(int len, float d = 1.0f) {
    if (len % 2 == 0) len += 1;  // Ensure odd length for symmetry
    int K = len / 2;  // Center index
    
    std::vector<float> h(len, 0.0f);
    
    // Center value
    h[K] = 1.0f / (4.0f * d * d);
    
    // Symmetric side lobes (only odd positions have non-zero values)
    for (int n = 1; n <= K; ++n) {
        if (n % 2 == 1) {
            float val = -1.0f / (float(PI) * float(PI) * n * n * d * d);
            h[K + n] = val;
            h[K - n] = val;
        }
    }
    
    return h;
}

/**
 * Apply Ramp filter to all projections in a sinogram (in-place convolution)
 * 
 * For each projection angle, convolve detector readings with ramp kernel.
 * This enhances high frequencies and suppresses low frequencies to prevent blurring.
 * 
 * @param sino     Sinogram data [n_angles, n_det] - modified in-place
 * @param n_angles Number of projection angles
 * @param n_det    Number of detector pixels
 * @param kernel   Ramp filter kernel
 */
static void filter_projections(float* __restrict sino,
    int n_angles, int n_det,
    const std::vector<float>& kernel) {
    const int K = int(kernel.size() / 2);
    const float* __restrict k_data = kernel.data();  // 优化：缓存指针

    // 并行化外层循环（角度循环）- 每个线程处理一个角度
    #pragma omp parallel
    {
        // 每个线程有自己的临时缓冲区，避免数据竞争
        std::vector<float> tmp(n_det);
        float* __restrict tmp_data = tmp.data();

        #pragma omp for schedule(static)
        for (int a = 0; a < n_angles; ++a) {
            float* __restrict row = sino + a * n_det;
            
            // 1D 卷积：计算每个检测器位置的滤波值
            for (int x = 0; x < n_det; ++x) {
                int k_start = std::max(-K, -x);
                int k_end   = std::min( K, n_det - 1 - x);

                double acc = 0.0f;

                for (int k = k_start; k <= k_end; ++k) {
                    acc += row[x + k] * k_data[K + k];
                }
                tmp_data[x] = acc;
            }
            
            // 将滤波后的值写回原数组
            #pragma omp simd
            for (int x = 0; x < n_det; ++x) {
                row[x] = tmp_data[x];
            }
        }
    }
}

void fbp_reconstruct_3d(
    float* __restrict sino_buffer,
    float* __restrict recon_buffer,
    int n_slices,
    int n_angles,
    int n_det,
    const std::vector<float>& angles_deg
) {
    const size_t slice_size = size_t(n_angles) * n_det;
    const size_t recon_size = size_t(n_det) * n_det;

    // ==== 采样间距（按你的真实数据设置） ====
    const double d_det = 1.0;  // 探测器像素间距

    // ---------- 预计算角度 & 梯形权 Δθ ----------
    const double deg2rad = double(PI) / 180.0;
    std::vector<double> cos_t(n_angles), sin_t(n_angles);
    #pragma omp parallel for schedule(static)
    for (int ai = 0; ai < n_angles; ++ai) {
        double t = double(angles_deg[ai]) * deg2rad;
        cos_t[ai]= std::cos(t);
        sin_t[ai]= std::sin(t);
    }

    // ---------- 几何中心 ----------
    const double cx = (n_det - 1) * 0.5;
    const double cy = (n_det - 1) * 0.5;
    const double t_center = (n_det - 1) * 0.5;
    const float t_half = (n_det - 1) * 0.5f;  // Detector offset to center
    const float scale = float(PI) / float(n_angles);  // Normalization factor from Radon inversion

    // ---------- STEP 1: Ramp 滤波（带窗 + 间距一致） ----------
    auto kernel = ramp_kernel(n_det | 1, float(d_det));
    #pragma omp parallel for schedule(static)
    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        float* sino_slice = sino_buffer + slice_id * slice_size;
        filter_projections(sino_slice, n_angles, n_det, kernel);
    }

    // 预先创建常量（避免每次循环重复创建）
    const int32x4_t zero_vec = vdupq_n_s32(0);
    const int32x4_t n_det_vec = vdupq_n_s32(n_det);
    const int32x4_t one_s32 = vdupq_n_s32(1);
    const float32x4_t zero_f = vdupq_n_f32(0.0f);
    const float32x4_t one_f = vdupq_n_f32(1.0f);

    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        // For each pixel in the reconstructed image
        const float* sino_slice = sino_buffer + slice_id * slice_size;
        float* recon_slice = recon_buffer + slice_id * recon_size;
        
        // 优化：改变循环顺序，先遍历y，再遍历角度
        // 这样可以更好地利用缓存，减少 recon_row 的访问次数
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < n_det; ++y) {
            double yr = y - cy;  // Y coordinate relative to image center
            float* recon_row = recon_slice + y * n_det;
            
            for (int ai = 0; ai < n_angles; ++ai) {
                double c = cos_t[ai];
                double s = sin_t[ai];
                const float* sino_row = sino_slice + ai * n_det;
                double last_u=-cx*c+yr*s+t_half;
                double inc=c;
                
                // // 预先计算增量向量（使用 double 精度）
                // float32x4_t inc_vec_lo = {0.0f, static_cast<float>(inc), static_cast<float>(2*inc), static_cast<float>(3*inc)};
                // float32x4_t inc_vec_hi = {static_cast<float>(4*inc), static_cast<float>(5*inc), static_cast<float>(6*inc), static_cast<float>(7*inc)};
 
                int x = 0;
                // 向量化处理，每次处理8个元素
                for (; x + 7 < n_det; x += 8) {
                    // 计算8个u值: last_u, last_u+inc, last_u+2*inc, ..., last_u+7*inc
                    // 这样写会先把相加需要的两个double转换为float，再相加，导致128个测试数据中有5个chu
                    // float32x4_t u_base_0 = vdupq_n_f32((float)last_u);
                    // float32x4_t u_vec_lo = vaddq_f32(u_base_0, inc_vec_lo);
                    // float32x4_t u_base_1 = vdupq_n_f32((float)(last_u+4*inc));
                    // float32x4_t u_vec_hi = vaddq_f32(u_base_1, inc_vec_lo);
                    
                    float32x4_t u_vec_lo = {static_cast<float>(last_u), static_cast<float>(last_u+inc), static_cast<float>(last_u+2*inc), static_cast<float>(last_u+3*inc)};
                    float32x4_t u_vec_hi = {static_cast<float>(last_u+4*inc), static_cast<float>(last_u+5*inc), static_cast<float>(last_u+6*inc), static_cast<float>(last_u+7*inc)};

                    // 计算整数部分 u0 (使用向下取整)
                    int32x4_t u0_lo = vcvtq_s32_f32(u_vec_lo);
                    int32x4_t u0_hi = vcvtq_s32_f32(u_vec_hi);
                    
                    // 计算小数部分 du = u - u0
                    float32x4_t du_lo = vsubq_f32(u_vec_lo, vcvtq_f32_s32(u0_lo));
                    float32x4_t du_hi = vsubq_f32(u_vec_hi, vcvtq_f32_s32(u0_hi));
                    
                    // 计算 u1 = u0 + 1
                    int32x4_t u1_lo = vaddq_s32(u0_lo, one_s32);
                    int32x4_t u1_hi = vaddq_s32(u0_hi, one_s32);
                    
                    // 边界检查 - 批量处理
                    uint32x4_t mask0_lo = vandq_u32(vcgeq_s32(u0_lo, zero_vec), vcltq_s32(u0_lo, n_det_vec));
                    uint32x4_t mask1_lo = vandq_u32(vcgeq_s32(u1_lo, zero_vec), vcltq_s32(u1_lo, n_det_vec));
                    uint32x4_t mask0_hi = vandq_u32(vcgeq_s32(u0_hi, zero_vec), vcltq_s32(u0_hi, n_det_vec));
                    uint32x4_t mask1_hi = vandq_u32(vcgeq_s32(u1_hi, zero_vec), vcltq_s32(u1_hi, n_det_vec));
                    
                    // Gather 操作 - 优化：减少临时数组和循环开销
                    int u0_arr[8], u1_arr[8];
                    vst1q_s32(u0_arr, u0_lo);
                    vst1q_s32(u0_arr + 4, u0_hi);
                    vst1q_s32(u1_arr, u1_lo);
                    vst1q_s32(u1_arr + 4, u1_hi);
                    
                    // 手动展开循环以提高效率
                    float sino0_data[8], sino1_data[8];
                    sino0_data[0] = sino_row[u0_arr[0]];
                    sino0_data[1] = sino_row[u0_arr[1]];
                    sino0_data[2] = sino_row[u0_arr[2]];
                    sino0_data[3] = sino_row[u0_arr[3]];
                    sino0_data[4] = sino_row[u0_arr[4]];
                    sino0_data[5] = sino_row[u0_arr[5]];
                    sino0_data[6] = sino_row[u0_arr[6]];
                    sino0_data[7] = sino_row[u0_arr[7]];
                    
                    sino1_data[0] = sino_row[u1_arr[0]];
                    sino1_data[1] = sino_row[u1_arr[1]];
                    sino1_data[2] = sino_row[u1_arr[2]];
                    sino1_data[3] = sino_row[u1_arr[3]];
                    sino1_data[4] = sino_row[u1_arr[4]];
                    sino1_data[5] = sino_row[u1_arr[5]];
                    sino1_data[6] = sino_row[u1_arr[6]];
                    sino1_data[7] = sino_row[u1_arr[7]];
                    
                    float32x4_t sino0_vec_lo = vld1q_f32(sino0_data);
                    float32x4_t sino0_vec_hi = vld1q_f32(sino0_data + 4);
                    float32x4_t sino1_vec_lo = vld1q_f32(sino1_data);
                    float32x4_t sino1_vec_hi = vld1q_f32(sino1_data + 4);
                    
                    // 插值计算 - 使用 FMA (fused multiply-add) 优化
                    // result = sino0 * (1-du) + sino1 * du
                    //        = sino0 - sino0*du + sino1*du
                    //        = sino0 + du*(sino1 - sino0)
                    
                    // lo 部分
                    float32x4_t weight0_lo = vsubq_f32(one_f, du_lo);
                    float32x4_t interp_lo = vmlaq_f32(vmulq_f32(sino0_vec_lo, weight0_lo), sino1_vec_lo, du_lo);
                    
                    // 应用边界掩码
                    float32x4_t sum0_lo = vbslq_f32(mask0_lo, vmulq_f32(sino0_vec_lo, weight0_lo), zero_f);
                    float32x4_t sum1_lo = vbslq_f32(mask1_lo, vmulq_f32(sino1_vec_lo, du_lo), zero_f);
                    float32x4_t result_lo = vaddq_f32(sum0_lo, sum1_lo);
                    
                    // hi 部分
                    float32x4_t weight0_hi = vsubq_f32(one_f, du_hi);
                    float32x4_t sum0_hi = vbslq_f32(mask0_hi, vmulq_f32(sino0_vec_hi, weight0_hi), zero_f);
                    float32x4_t sum1_hi = vbslq_f32(mask1_hi, vmulq_f32(sino1_vec_hi, du_hi), zero_f);
                    float32x4_t result_hi = vaddq_f32(sum0_hi, sum1_hi);
                    
                    // 加载当前recon_row的值并累加
                    float32x4_t recon_lo = vld1q_f32(recon_row + x);
                    float32x4_t recon_hi = vld1q_f32(recon_row + x + 4);
                    
                    recon_lo = vaddq_f32(recon_lo, result_lo);
                    recon_hi = vaddq_f32(recon_hi, result_hi);
                    
                    vst1q_f32(recon_row + x, recon_lo);
                    vst1q_f32(recon_row + x + 4, recon_hi);
                    
                    // 更新last_u
                    last_u += inc * 8;
                }
                
                // 处理剩余的元素
                for (; x < n_det; ++x) {
                    double u = last_u;
                    last_u = u + inc;
                    
                    int u0 = int(u);
                    float du = u - u0;
                    int u1 = u0 + 1;
                    
                    float sum0 = (u0 >= 0 && u0 < n_det) ? sino_row[u0]*(1.0f - du) : 0.0f;
                    float sum1 = (u1 >= 0 && u1 < n_det) ? sino_row[u1]*du : 0.0f;
                    
                    recon_row[x] += sum0 + sum1;
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < n_det; ++y) {
            float* __restrict row = recon_slice + y * n_det;
            #pragma omp simd
            for (int x = 0; x < n_det; ++x) row[x] *= scale;
        } 
    }
   
}