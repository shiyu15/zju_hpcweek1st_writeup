# RISC-V 大模型推理算子性能优化挑战

## 赛题背景

近年来，大语言模型（Large Language Model, LLM）在机器翻译、问答系统和代码生成等任务中展现出卓越的能力。然而，随着模型规模的不断扩大，推理阶段的算力消耗和延迟问题日益凸显。

在大模型推理过程中，矩阵乘法（GEMM/GEMV）等线性代数运算占据了绝大部分计算开销。如何在新兴的 RISC-V 架构生态中，充分利用其指令集特性，设计高效的算子实现，为大语言模型提供高性能、低功耗的推理服务，已成为产业界和学术界共同关注的重要课题。

在本赛题中，你将自己亲身体验在一个真实的大模型推理框架中，优化量化矩阵乘法算子的全过程。

## 赛题任务

### 任务概述

本赛题基于 [llama.cpp](https://github.com/ggml-project/llama.cpp) 推理框架，使用 DeepSeek-R1-Distill-Qwen-1.5B 模型，采用权重 4-bit 量化、激活 8-bit 量化的配置进行推理。

你需要优化该推理过程中的核心计算瓶颈——量化矩阵乘法算子。赛题已将该算子抽象为独立的共享库接口，位于 `qmatmul.c` 文件中。你的任务是优化该接口的算子实现，提升整体推理性能。赛题提供了参考实现，该版本能够正确完成计算，但性能有待提升。你可以基于此版本进行改进。

### 性能评估

赛题主要关注以下两个维度的性能指标：

- **Prefilling 阶段吞吐量**
- **Decoding 阶段吞吐量**

这两个阶段分别对应模型推理的不同环节，优化方案需要在保证正确性的前提下，提升整体性能。

性能输出示例如下：

```text
llama_perf_sampler_print:    sampling time =     446.03 ms /   281 runs   (    1.59 ms per token,   630.00 tokens per second)
llama_perf_context_print:        load time =    4200.36 ms
llama_perf_context_print: prompt eval time =   14612.40 ms /    15 tokens (  974.16 ms per token,     1.03 tokens per second)
llama_perf_context_print:        eval time =  292949.38 ms /   265 runs   ( 1105.47 ms per token,     0.90 tokens per second)
llama_perf_context_print:       total time =  308324.01 ms /   280 tokens
llama_perf_context_print:    graphs reused =        263
```

其中，1.03 tokens per second 对应 Prefilling 阶段吞吐量，0.90 tokens per second 对应 Decoding 阶段吞吐量。

## 赛题环境

- riscv 分区下共有 6 个调试节点，每个节点有 8 个 CPU 核心。你可以通过 `srun` / `sbatch` 等命令提交任务，最多使用 1 个节点的 8 个核心，单任务限时 5 分钟。
- 你可以使用 `gcc` / `clang` / 其他编译器，编译选项不限，但请确保编译结果能在 RISC-V 架构上正确运行。
- 赛题使用 CMake 进行构建，相关配置已在 `CMakeLists.txt` 中给出。你可以使用提供的脚本 `run.sh` 进行编译和运行。
- 赛题所需的所有依赖均已预装在系统中，在公共目录 `/hpcweek/rvllm/` 下
  - `llama.cpp`：llama.cpp 执行文件
  - `model/`：DeepSeek-R1-Distill-Qwen-1.5B 模型文件
  - `ggml/`：用于编译的 ggml 库文件

## 赛题要求

### 1. 正确性要求

- 测评系统将对提交的算子实现进行正确性校验
- 未通过正确性校验的提交计为 0 分
- 你可以通过比较模型输出文本与参考输出文本是否一致，来验证实现的正确性

### 2. 修改范围限制

- 仅允许修改 `qmatmul.c` 和 `CMakeLists.txt` 文件
- 修改时必须保持接口与原有实现一致，确保不影响整体推理流程

### 3. 评分标准

- **Prefilling 阶段吞吐量提升得分**（权重 50%）
- **Decoding 阶段吞吐量提升得分**（权重 50%）
- 各阶段得分基于相对于参考实现的性能提升比例计算
- 最终得分为两个阶段得分的加权平均值

### 4. 诚信守则

- 参赛者必须独立完成优化工作，禁止抄袭或协作。我们将对提交的代码进行查重，一经发现本题分数计为 0 分。

## 参考资料

- [RISC-V 向量扩展（RVV）官方规范](https://github.com/riscv/riscv-v-spec)
- [llama.cpp 项目](https://github.com/ggerganov/llama.cpp)
- [HPC101 RISC-V 向量化实验指导](https://hpc101.zjusct.io/lab/Lab2.5-RISC-V/)