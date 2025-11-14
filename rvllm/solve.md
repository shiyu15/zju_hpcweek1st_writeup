# rvllm
这道题的推理框架是llama.cpp，我上次看见它还是两三年前。那时候作者Georgi Gerganov独自写了这个框架的消息被ai三大顶会、知乎广泛报道。今年一搜索发现已经89k的star了。相比vllm，它更适合在一些边缘计算的设备上部署。

这道题也很有意思，这个推理框架还很智能，我使用`Ofast`的优化等级，直接报错，说不允许这样的优化。开发者竟然连这也能注意到，非常有实力。
```cpp
hpcweek/rvllm/ggml/src/ggml-cpu/vec.h:965:2: error: #error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
  965 | #error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
      |  ^~~~~
/hpcweek/rvllm/ggml/src/ggml-cpu/vec.h:966:2: error: #error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
  966 | #error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
      |  ^~~~~
```


## 优化
### 1.使用riscv汇编
看到一个算子是矩阵乘法的，一个算子是向量点积的，第一个算子会用到第二个算子的实现。
用riscv写了一版第二个向量点积算子，差不多有190分了。暑假的超算小学期有一些写riscv汇编的[教程](https://hpc101.zjusct.io/lab/Lab2.5-RISC-V/)，很有用。

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

之后我费力去进行优化，最终也差不多是190分，不是很懂还能怎么提高性能。

### 2.优化尝试
我使用printf打印出第一个算子`ggml_compute_forward_mul_mat_one_chunk`的一些入参，看看有没有可以缓存的，或者可以针对一些特定的路径进行优化。但是结果没有好的。我现在这个算子的实现应该和原版性能没有差别。
