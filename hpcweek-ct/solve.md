# ct扫描优化
## 背景
ct扫描也是并行计算的经典样例了，在今年出版的《大规模并行处理器程序设计》中就有针对ct成像迭代时计算的优化方法。不过这本书讨论的是cuda上面的优化，和这道题目的cpu优化关系比较小。

## 优化方法

### 1.调整循环顺序
原始的四层循环的顺序是`slice_id → x → y → ai`,先循环图片id，再循环位置坐标，再循环每个角度
```CPP
for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
    // Pointers to input (filtered sinogram) and output (reconstructed image)
    const float* sino_slice = sino_buffer + slice_id * slice_size;
    float* recon_slice = recon_buffer + slice_id * recon_size;
    
    // For each pixel in the reconstructed image
    for (int y = 0; y < n_det; ++y) {
        float yr = y - cy;  // Y coordinate relative to image center
        float* recon_row = recon_slice + y * n_det;
        
        for (int x = 0; x < n_det; ++x) {
            float xr = x - cx;  // X coordinate relative to image center
            float acc = 0.0f;

            for (int ai = 0; ai < n_angles; ++ai) {
                ...
            }
            
            recon_row[x] = acc * scale;
        }
    }
}
```

我们可以把这个循环的顺序调整到 `slice_id → y → ai → x` ，从原来的对每个`(x,y)` 遍历所有角度 ai，改成每个 (y,ai) 遍历所有 x，recon_row 连续访问，方便进行simd的修改。我也尝试过用`slice_id → ai → y → x`的顺序，相比原始顺序有一定提速，但不如这个顺序快。
```cpp
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
                int x = 0;
                // 向量化处理，每次处理8个元素
                for (; x + 7 < n_det; x += 8) {
                    ...
                    
                    vst1q_f32(recon_row + x, recon_lo);
                    vst1q_f32(recon_row + x + 4, recon_hi);
                }
            }
        }

```

### 2.用加法优化乘法
原先的程序每次循环，最内部的ai每次改变都要做两次乘法。非常消耗时间。
```cpp
        // For each pixel in the reconstructed image
        for (int y = 0; y < n_det; ++y) {
            float yr = y - cy;  // Y coordinate relative to image center
            float* recon_row = recon_slice + y * n_det;
            
            for (int x = 0; x < n_det; ++x) {
                float xr = x - cx;  // X coordinate relative to image center
                float acc = 0.0f;
                
                // Accumulate contributions from all projection angles
                for (int ai = 0; ai < n_angles; ++ai) {
                    float c = cos_theta[ai];
                    float s = sin_theta[ai];
                    
                    // Radon transform: t = x*cos(θ) + y*sin(θ)
                    // This is the position where pixel (x,y) projects onto detector at angle θ
                    float t = xr * c + yr * s;
                    
                    // Convert to detector coordinate (0 to n_det-1)
                    float u = t + t_half;
                    
                    // Integer and fractional parts for interpolation
                    int u0 = int(u);
                    float du = u - u0;
                    int u1 = u0 + 1;
                }
            }
        }
```
我们在调整循环顺序，把ai放到最外层之后就可以把这个乘的操作改为加的操作。
```cpp
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
                int x = 0;

                for (; x < n_det; x ++) {
                    double t = last_u +inc;
                    ...
                }
            }
        }
```
这里也需要注意精度问题，如果last_u和inc用float保存的话，多次相加造成的误差会让程序过不了精度测试，因此需要用精度较高的double来储存。

### 3.使用simd
使用simd一次处理8个x，提高计算的速度。不止计算的部分可以simd，也可以将simd用在条件判断。可以看到我的simd中每次取数据都要做加法和乘法，原来想着优化一下。但是如果把{0，inc*1, inc*2, inc*3}储存进simd向量来优化掉每次要做的乘法，会达不到精度要求。
我
```cpp
for (int ai = 0; ai < n_angles; ++ai) {
    double c = cos_t[ai];
    double s = sin_t[ai];
    const float* sino_row = sino_slice + ai * n_det;
    double last_u=-cx*c+yr*s+t_half;
    double inc=c;

    int x = 0;
    // 向量化处理，每次处理8个元素
    for (; x + 7 < n_det; x += 8) {
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
}
```

### 4.比较输出结果和参考结果
修改完fbp.cpp之后，我们想要检查一下当前的程序满不满足精度要求，我们可以用sha256sum来检查一下和之前未修改前的程序有没有输出上的错误。如果sha256sum输出完全相同，则没有错误。如果输出不同，则可能有错误，可以传到oj上测试一下。
