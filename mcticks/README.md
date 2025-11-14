# 题目背景

在遥远的、由代码和像素构成的宇宙深处，存在着一个古老而骄傲的文明——**火把星球**。

这里的居民，自称“火把星人”，是一种由发光像素点和永不熄灭的火焰之魂组成的奇妙生命体。他们不耕田，不织布，毕生只钻研两门至高无上的科学：“如何让光源闪得更花里胡哨”以及“如何让一桶水淹了整个服务器”。

经过无数个游戏刻的潜心研究，火把星的科学天团终于取得了突破性进展，掌握了改变世界底层规则的力量——**光源方块与流体方块的终极操控术**！他们兴奋地搓着手上的火焰，打造了三款足以令主世界CPU哭泣的传说级武器：

*   **火把光粒加农炮**：发射的不是炮弹，是海量的光源更新！一炮下去，目标区域将陷入疯狂的光影计算地狱，TPS直接从20跌到0。
    
*   **水与火之歌交响发生器**：同时召唤奔腾的流水与燃烧的烈焰，让流体与光源的更新在方块间上演一场史诗级（且卡顿）的华尔兹，用极致的逻辑冲突考验着任何物理引擎的血压。
    
*   **闪烁火把领域展开**：创造一个范围巨大的领域，领域内的每一个火把都以不同的频率疯狂闪烁，营造出最炫酷的迪厅氛围，顺便让光照计算彻底崩溃。
    

野心勃勃的火把星元老院一拍即合：“征服主世界，让所有玩家在掉帧与卡顿中臣服于我们的光芒！”

“先选一个服务器进行攻占吧！” 

### 与此同时，另外一边

ZJUSCT 有一个 MC 服务器，最近不知道为什么似乎出了点问题，似乎有人在同时放置大量高空流水、产生大量光源更新，让服务器压力很大，tps极低。为了保护游戏体验，star, chenhz 和 jrguo 决定手搓一个服务器，在正式开工之前他们制作了一个只涉及光照和流体行为的 demo，不过性能不能很好地支撑玩家游玩。在他们准备开始优化的时候，他们因为每天熬大夜病倒了，于是他们把这个优化任务交给了你。

# 题目描述

本题实现了一个基本的Minecraft服务器，适用于 Java Edition 1.21.4。基准代码参考 [MC WIKI](https://minecraft.wiki/) 中的功能描述，在Java Edition 的基础上进行了一些更适用于 HPC 的调整。大致逻辑是在一个循环（称为 Tick）里执行了光照更新、流体更新等的行为，具体的代码理解都以注释的形式给出，你需要在**不改变**代码执行结果的情况下，尽可能的对代码进行优化。本赛题在正确性得到保障的情况下，支持一切形式的优化手段。

# 使用指南

## 构建

### 构建cpp模拟库。

```bash
git clone https://git.zju.edu.cn/zjusct/mcticks.git

cd mcticks

cmake -B build
cmake --build build --parallel
```

### 构建 go 适配器

```bash
# 自行安装 golang 1.25 版本 https://golang.google.cn/dl/
cd go
go env -w GOPROXY=https://goproxy.cn,direct
go build

cd ..
```

你也可以选择直接下载我们构建好的二进制文件：

*   [macos-arm64](index.assets/macos-arm64)
*   [linux-amd64](index.assets/linux-amd64)
*   [windows-amd64](index.assets/windows-amd64.exe)


同时位于本仓库的 `index.assets/` 目录下。

为了和下文使用一致，你可以将其重命名为 `go-mc`，并放到 `go/` 目录下。
    
## 开始使用

所有脚本中的路径均相对于启动可执行文件时的工作目录。

以下假设为在项目根目录下执行
```bash
# macos
# 开启服务器
DYLD_LIBRARY_PATH="./build" ./go/go-mc 

# 运行 torch-mars.mcsh 脚本
DYLD_LIBRARY_PATH="./build" ./go/go-mc run scripts/torch-mars.mcsh 

# 比对运行结果
DYLD_LIBRARY_PATH="./build" ./go/go-mc compare torch_mars.mccs ref/torch_mars.mccs

# linux 下把 DYLD_LIBRARY_PATH 改成 LD_LIBRARY_PATH，即：
LD_LIBRARY_PATH="./build" ./go/go-mc

# windows 可用 msys2 进行构建，把编译好的DLL文件放到和可执行文件同一目录下。具体用法请各位同学自行探索。更推荐直接使用WSL。
```

默认会开启在 0.0.0.0:25565 的 Minecraft 服务器，可在 config.json 里进行配置。可以使用 `Minecraft Java Edition 1.21.4` 进行连接，支持多人游戏，配合光影食用更佳。

## 执行预制动画脚本

可以在游戏中输入命令 /script filename 来执行预制的动画脚本。同时，你也可以在程序的`stdin`输入命令，也可以使用命令行 `go-mc run filename` 来执行脚本。 命令实现位于 [world.go](go/world/world.go) 文件中，你可以参考该文件来理解这些命令的用法。

## 测评

我们使用 `judge_create` 命令来进行测评，命令格式如下：

```bash
judge_create world_series.mccs <total_ticks> <save interval> <save_chunk_start_x> <save_chunk_start_z> <save_chunk_end_x> <save_chunk_end_y> judge_tps.json
```

例如： `judge_create world_series.mccs 80 3 -8 -8 8 8 judge_tps.json` 其会在当前世界中向前运行 80 个 tick，每 3 个 tick 保存一次区块，保存到 `world_series.mccs` 中，保存区域为从 (-8, -8) 到 (8, 8) 的区块，最终将测评性能结果保存到 `judge_tps.json` 文件中。  `judge_tps.json` 为可选参数，如果不提供则不会输出测评结果文件。

计时区为：**从测评开始到最后一个tick结束的墙钟时间**, 请注意这与 _tps_ 命令的输出不相同。  示例测评结果文件如下

```json
{"total_ticks":8000,"total_time_seconds":0.631371334}
```

这里的 `total_ticks/total_time_seconds=12670.83183729` 即为 _测评标准TPS_。

请注意，为了排除保存区块的IO开销对测评结果的影响，在OJ上运行时，我们会先测试正确性，然后进行性能测评。
此时，我们会先运行一次 `judge_create` 命令来确保结果正确，然后再运行一次 `judge_create` 命令来进行性能测评，期间不会保存任何区块数据，例如`judge_create world_series.mccs 80 80 -8 -8 8 8 judge_tps.json`。

我们会将保存的每个区块与标准答案进行对比，区块的方块数据，光照数据必须完全一致。参考命令

```bash
go-mc compare fileA.mccs fileB.mccs
```

在调试过程中，你可以使用 `judge_compare` 命令来进行实时对比，命令格式如下：

```bash
judge_compare world_series.mccs <total_ticks> <save interval> <save_chunk_start_x> <save_chunk_start_z> <save_chunk_end_x> <save_chunk_end_z>
```
### 提交至OJ

OJ 会收取以下目录结构，请在本地严格排列后使用 `选择文件夹` 功能选择 `mcticks` 文件夹进行提交：

```
mcticks/
├── world/
├── util/
├── CMakeLists.txt
```

最多上传30个文件，总大小不超过1MB。  
你也可以尝试 https://github.com/ZJUSCT/CSOJ-cli 进行命令行提交。

### 评分标准

该题目使用打榜评分的方法，总分250分，在性能达到起评线且三个世界结果都正确时开始得分。OJ 测评时使用 **x86** 分区的 **26** 个物理核心。

我们会给出 3 个动画脚本，位于 `scripts/` 目录下，分别为：

*   `torch-mars.mcsh`: 火把星
    
*   `light-boom.mcsh`: 超平坦闪烁光照
    
*   `mountain-water.mcsh`: 高山流水
    

最后性能得分为三个世界的加权平均 _测评标准TPS_。

baseline测评参考性能，世界权重和起评线为:

| 世界名称 | baseline参考性能（测评标准TPS） | 权重 |
| -------- | ------------------------------ | ---- |
| mountain-water.mcsh | `0.511` | `0.4` |
| light-boom.mcsh | `0.367` | `0.4` |
| torch-mars.mcsh| `4.231` | `0.2` |

加权起评线: `3.0`

请注意，视同学们的答题情况，本题的评分标准仍有可能进行适当微调。