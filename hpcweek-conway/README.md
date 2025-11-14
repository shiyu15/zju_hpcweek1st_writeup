## 题目背景

Conway's Game of Life（康威生命游戏）是由数学家 John Horton Conway 于 1970 年提出的经典细胞自动机。 

它在一个二维无限网格上演化，每个格子可能处于「存活」或「死亡」两种状态，下一时刻的状态仅取决于周围八个邻居格子的情况。  

具体规则如下：

1. 存活细胞若周围存活邻居数为 2 或 3，则继续存活；否则死亡。

4. 死亡细胞若周围存活邻居数恰好为 3，则会复活；否则仍然死亡。

尽管规则非常简单，但生命游戏展现出了丰富复杂的动态行为，被广泛应用于复杂系统、理论计算机科学和人工生命等领域。 

在 HPC 相关实验和竞赛中，生命游戏常被用作测试并行优化和大规模模拟的经典问题。

## 任务说明

给出[生命游戏模拟程序](https://git.zju.edu.cn/zjusct/hpcweek-conway)：给定一个初始的有限宇宙（二维网格状态）和迭代次数，计算出若干次迭代后的最终状态。你的任务是优化这个模拟过程。

你可以在给出的代码框架中自由修改 `conway.py` 中指定区域及 `NG.cpp` 中内容，提交时我们会将区域内代码替换到评测程序中。

对于 python 实现，你可以使用 `requirements.txt` 中给出的库；

## 输入与输出

参考 `Expand()` 函数。其中返回的 `grid` 只用保证细胞的整体形状一致即可，四个方向没有活细胞的空行对最终正确性判定没有影响。

python 中 `Next_Generation_Ref()` 函数的返回参数是 `grid, (dy, dx)`，其中 `(dy, dx)` 用以表示可视化数组偏移，实现优化时可以忽略。

**注意：**

- 评测不会直接读取或比对中间过程，只检查最终结果。

- 正确性优先，其次是运行效率。

## 数据范围

初始数据边长、迭代次数数量级为 1e2

## 可视化

我们提供了可视化选项，供你查看实际演化过程、测试自己实现的计算函数正确性。

为了使用可视化模块，你需要：

自己在 python 或 cpp 中实现 Next_Generation 函数，并替换 `visualize.py` 中 `class World` 内 `evolve()` 函数里的 `Next_Generation_Ref` 函数。对于 cpp，目前只导出了 `expand_cpp` 函数。你需要添加 `next_generation_cpp` 的导出逻辑。

整体可视化逻辑通过 python asyncio 的协程编程模型实现，你可以进行以下操作：

+ ↑↓←→: 控制视角移动

+ +/-: 增减演化间隔时间

+ Space (空格): 暂停自动演化

+ J: 在演化暂停后，按 J 进行单步演化

另外，请注意可视化主要用于查看生命游戏演化规律/测试你自己逻辑的正确性，实际测试性能与整个可视化模块无关；同时，在你本地实现时，如有需要，可以自由修改可视化模块的逻辑。

## 运行方法

```shell
usage: main.py [-h] [-V] [-I ITER] (-F FILE | -S HEIGHT WIDTH)

Conway's Game of Life simulation.

options:
  -h, --help            show this help message and exit
  -V, --visualize       Enable visualization.
  -I ITER, --iter ITER  Number of iterations.
  -F FILE, --file FILE  Path to an RTE file to load the initial grid.
  -S HEIGHT WIDTH, --size HEIGHT WIDTH
                        Height and width for a random grid.
```

## 提交

请在 oj 评测中提交 `NG.cpp` 和 `conway.py` 两个文件。