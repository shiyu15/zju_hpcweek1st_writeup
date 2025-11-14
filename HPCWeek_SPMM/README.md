# 稀疏矩阵乘优化

## 简介

稀疏矩阵-稠密矩阵乘法`SpMM`是科学计算和工程模拟中的基础性计算内核，其计算形式等价于矩阵乘法，是通用矩阵乘法`GEMM`的特化版本

```text
C = A × B
```

其中：

- **A** 是 M×K 的稀疏矩阵（通常非零元素比例 <5%），采用 CSR 格式存储
- **B** 是 K×N 的稠密矩阵  
- **C** 是 M×N 的结果矩阵

### 稀疏矩阵介绍

稀疏矩阵是指大部分元素为零的矩阵。在实际应用中，很多矩阵的非零元素占比不到5%，甚至更少。

#### 示例矩阵（6×6）

```text
[ 2.0   0.0   0.0   3.0   0.0   0.0 ]
[ 0.0   4.0   0.0   0.0   0.0   1.0 ]
[ 0.0   0.0   0.0   0.0   0.0   0.0 ]
[ 1.0   0.0   0.0   5.0   0.0   0.0 ]
[ 0.0   0.0   2.0   0.0   6.0   0.0 ]
[ 0.0   0.0   0.0   0.0   0.0   7.0 ]
```

这个矩阵有36个位置，但只有8个非零元素，稀疏度为78%。

#### CSR格式

如果按照传统的二维数组存储这个6×6矩阵，需要存储36个浮点数。但实际上只有8个有用的数据，其余28个都是0，造成了巨大的存储浪费和计算浪费。

`CSR(Compressed Sparse Row)`是最常用的稀疏矩阵存储格式之一。它用三个一维数组来表示稀疏矩阵：

##### 存储格式介绍

`values`：存储所有非零元素的值

`col_idx`：存储每个非零元素对应的列索引  

`row_ptr`：存储每行第一个非零元素在values数组中的位置

##### 访问矩阵元素

要访问第i行的所有非零元素：

```cpp
for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
    int column = col_idx[j];     // 列索引
    float value = values[j];     // 对应的值
    // 矩阵A[i][column] = value
}
```

## 优化任务

### 项目结构

```text
.
├── build_and_run.sh
├── CMakeLists.txt
├── data
│   ├── heart1.mtx
│   ├── orani678.mtx
│   ├── psmigr_1.mtx
│   ├── psmigr_2_block_0_0.mtx
│   └── psmigr_3_block_0_0.mtx
├── include
│   ├── csr_matrix.h
│   ├── matrix_utils.h
│   ├── spmm_opt.h
│   ├── spmm_ref.h
│   └── test_case.h
├── main.cpp
├── README.md
├── run_test.sh
└── src
    ├── spmm_opt.cpp
    ├── spmm_ref.cpp
    └── test_case.cpp
```

**优化目标**：优化 `src/spmm_opt.cpp` 中的 `spmm` 实现，其中 `ref` 是参考实现，运行`./run_test.sh` 中的七组测试数据(其中5组为真实矩阵 2组为随机生成的矩阵)，根据`GFLOPS`的平均值的排名获得分数

### 测试运行

```bash
mkdir build
cd build
cmake ../
make
```

会生成一个`./build/spmm`可执行文件 

```bash
./spmm -m 1024 -n 1024 -k 1024 # 测试运行 (-m / -n / -k 表示指定的矩阵形状大小)
./run_test.sh #运行脚本
```

### 评测

根据文档提交`OJ`的方式，**你只需要提交spmm_opt.cpp 和 CMakeLists.txt**， 我们会在OJ上对你的代码进行评测和给分。