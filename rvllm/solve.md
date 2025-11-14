看到一个算子是矩阵乘法的，一个算子是向量点积的，第一个算子会用到第二个算子的实现。

用riscv写了一版第二个向量点积算子，差不多有189分了

```cpp
    // 处理每个 block (每个 block 有 32 个元素)
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
```