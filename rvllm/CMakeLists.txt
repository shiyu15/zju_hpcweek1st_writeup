
cmake_minimum_required(VERSION 3.16)
project(HPCWeek_rvLLM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(GGML_DIR /hpcweek/rvllm/ggml)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
)

add_library(qmatmul SHARED
    src/qmatmul.c src/mul_mat_compute.c src/type_traits.c
)

target_compile_options(
    qmatmul PRIVATE
    -Wall
    -Wextra
    -fPIC
    -O3
    -Wno-unused-function
    -march=rv64gv -mabi=lp64d
    
    # OpenMP 并行优化
    -fopenmp
    
    # 循环优化
    -funroll-loops              # 循环展开优化
    -fprefetch-loop-arrays      # 循环数组预取
    -ftree-loop-distribution    # 循环分布优化
    -ftree-loop-vectorize       # 循环向量化
    
    # 向量化优化
    -ftree-vectorize            # 自动向量化
    -ftree-slp-vectorize        # SLP 向量化
    
    # 内联和函数优化
    -finline-functions          # 内联函数优化
    -finline-limit=600          # 提高内联限制
    
    # 寄存器和内存优化
    -fomit-frame-pointer        # 省略帧指针以释放寄存器
    -fno-strict-aliasing        # 允许更灵活的指针优化
    
    # 对齐优化
    -falign-functions=32        # 函数对齐优化
    -falign-loops=32            # 循环对齐优化
    
    # 数学优化（安全版本，不使用 fast-math）
    -fno-math-errno             # 不设置数学错误标志
    -fno-trapping-math          # 假设浮点运算不会产生陷阱
    
    # 其他高级优化
    -fgcse-after-reload         # 重新加载后的公共子表达式消除
    -fipa-pta                   # 过程间指针分析
    -fmerge-all-constants       # 合并所有常量
)

# 链接选项（必须包含 OpenMP）
target_link_options(qmatmul PRIVATE
    -fopenmp                # OpenMP 链接（必须！）
)
