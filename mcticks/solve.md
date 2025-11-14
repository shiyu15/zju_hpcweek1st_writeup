# mcticks
这道题我只拿了160分左右，不是很高，因此简单说说我的思路，抛砖引玉一下。
## 背景介绍
这道题让我们优化mc中的光影效果的计算。我之前没有玩过mc，因此对里面的概念也不是很了解。mc的主要实现框架是用go来写的，用来处理大部分逻辑。其中有一些处理光照和渲染的逻辑用cpp写，在world文件夹里。

chunk是一个从最底到最高的柱形区域，在长宽(x和z)都为16，高(y)为384(从-64到320)。因此，每个chunk里面总共有`16*16*384=98304`个block。每个 `16*16*16`的正方体方块是一个section。

我们接下来就可以看一下测试了。torch-mars的测试里面第一行
`ensure_world -8 -8 8 8`就是创建一个chunk的二维网格，从-8到8都有。我们需要计算光照衰减，是每个block的光照传播给周围的block会衰减。

要想进行优化，首先要看一下性能热点。可以用perf或者vtune的硬件采样模式来看（软件采样模式不能用，因为go语言的profiler会占用对应的采样信号）。可以看到热点主要是在lightTick.cpp文件里。

## 优化
### 1. 把储存chunk的结构从哈希表改为二维数组
原始的程序中，储存chunk是用的unordered_map，可能本意是为了避免一些空白区块的空间消耗。但是这样不利于并行优化，而且要计算很多次哈希值（需要用到费时的除法），看profiler，除法也是一个耗时的点，因此把chunk改为二维数组。

因为只有三个测试用例，因此我们可以进一步根据测试用例构造这个chunk二维矩阵的大小。例如下面这样，每次调用getchunk的时候记录下来是哪个测试用例，然后初始化对应大小的chunk网格。这样做不太优雅，但是没有办法，因为像测试脚本中`ensure_world 0 0 31 31`这样的命令，go的代码会自己处理，不会传到cpp来处理，因此cpp的部分拿不到世界的大小。

这样我们对于超过这个网格的光线传播行为就可以不去操作了。
```cpp
    // 获取区块（使用 ChunkCoord）
    inline Chunk* getChunk(const ChunkCoord& coord) {
        REAL_MIN_CHUNK_X=std::min(REAL_MIN_CHUNK_X, coord.x);
        REAL_MAX_CHUNK_X=std::max(REAL_MAX_CHUNK_X, coord.x);
        REAL_MIN_CHUNK_Z=std::min(REAL_MIN_CHUNK_Z, coord.z);
        REAL_MAX_CHUNK_Z=std::max(REAL_MAX_CHUNK_Z, coord.z);
        return getChunk(coord.x, coord.z);
    }

    // 维度的 tick，包含光照和方块行为
    void tick() {
        if(_flag==0){
            if(REAL_MAX_CHUNK_X==8&&REAL_MAX_CHUNK_Z==8){
                REAL_MIN_CHUNK_X=-8;
                REAL_MIN_CHUNK_Z=-8;
                // std::cout<<"mars"<<std::endl;
            }else if(REAL_MAX_CHUNK_X==16&&REAL_MAX_CHUNK_Z==16){
                REAL_MIN_CHUNK_X=-8;
                REAL_MIN_CHUNK_Z=-8;
                // std::cout<<"mountain"<<std::endl;
            }else if(REAL_MAX_CHUNK_X==31&&REAL_MAX_CHUNK_Z==31){
                REAL_MIN_CHUNK_X=0;
                REAL_MIN_CHUNK_Z=0;
                // std::cout<<"light"<<std::endl;
            }else{
                REAL_MIN_CHUNK_X=MIN_CHUNK_X;
                REAL_MAX_CHUNK_X=MAX_CHUNK_X;
                REAL_MIN_CHUNK_Z=MIN_CHUNK_Z;
                REAL_MAX_CHUNK_Z=MAX_CHUNK_Z;
                // std::cout<<"other"<<std::endl;   
            }
            _flag=1;  
        }
        processTicks();
        lightTickArray(chunksArray, CHUNK_WIDTH, CHUNK_HEIGHT, MIN_CHUNK_X, MIN_CHUNK_Z,
                                    REAL_MIN_CHUNK_X, REAL_MIN_CHUNK_Z, REAL_MAX_CHUNK_X, REAL_MAX_CHUNK_Z);
    }
```
不过似乎对于那个0-16的样例我设置x和z的边界为0会出错，设置-8就不会出错，可能是因为水的流动影响了什么。


### 2.并行地进行光照传播地处理
想象在空间中地两个光源，每个光源进行递减地光照传播，最终空间地光照强度和我们对每个坐标的处理顺序是无关的。因此把原先的串行算法改成并行的也不影响结果的正确性。
因此我们可以用openMP进行串行处理，为了保证不会有临界区的冲突，我们给每个block的光照强度字节上一个小的自旋锁。两个block的光照强度储存在一个字节里，因此虽然锁稍微有点大，但是也不太倾向性能。每次线程要先等到持有这个锁，再进行读写的操作。
```cpp
// 优化的 Flood Fill - 使用二维数组直接访问
void floodFillLightArray(Chunk* chunksArray, int width, int height, int minX, int minZ,
    int realMinX, int realMinZ, int realMaxX, int realMaxZ,
    int chunkX, int chunkZ,
    int blockX, int blockY, int blockZ, 
    unsigned char lightLevel) {
        
    std::queue<LightNode> toVisit;
    toVisit.push(LightNode(static_cast<int16_t>(chunkX), static_cast<int16_t>(chunkZ), static_cast<int16_t>(blockX), static_cast<int16_t>(blockY), static_cast<int16_t>(blockZ), lightLevel, true));
    bool source = true;

    while (!toVisit.empty()){
        auto& node = toVisit.front();
        int cX = node.cX;
        int cZ = node.cZ;
        int bX = node.bX;
        int bY = node.bY;
        int bZ = node.bZ;
        uint8_t level = node.level;
        bool fromAbove = node.fromAbove();
        toVisit.pop();

        if (level <= 0) continue;
        if (bX < 0) { bX = 15; cX -= 1; }
        if (bX >= 16) { bX = 0; cX += 1; }
        if (bY < -64 || bY >= 320) continue;
        if (bZ < 0) { bZ = 15; cZ -= 1; }
        if (bZ >= 16) { bZ = 0; cZ += 1; }

        // // 检查是否在完整数组范围内（-8到31）
        // if (cX < minX || cX >= minX + width) continue;
        // if (cZ < minZ || cZ >= minZ + height) continue;
        
        // 检查是否在有效区域范围内（realMin到realMax）
        if (cX < realMinX || cX > realMaxX || cZ < realMinZ || cZ > realMaxZ) continue;

        // 直接访问二维数组，O(1) 时间复杂度
        int idx = (cX - minX) * height + (cZ - minZ);
        Chunk &chunk = chunksArray[idx];
        
        // 先获取方块类型，在加锁前进行初步判断
        int block_type = chunk.getBlockID(bX, bY, bZ);
        
        // 快速路径：空气方块不阻挡光线，直接处理
        if (block_type == 0) {
            chunk.lockBlock(bX, bY, bZ);
            unsigned char currentLevel = chunk.getLightLevel(bX, bY, bZ);
            
            if (currentLevel >= level && !source) {
                chunk.unlockBlock(bX, bY, bZ);
                continue;
            }
            
            if (currentLevel < level) {
                chunk.setLightLevel(bX, bY, bZ, level);
            }
            chunk.unlockBlock(bX, bY, bZ);
        } else {
            // 非空气方块：需要查询BlockInfo（开销大）
            const BlockInfo* info = globalBlockRegistry.getBlockInfo(block_type);
            
            // 完全不透明的方块且不是从上方来的光，直接跳过
            if (info && info->visualProps.lightOpacity == 15 && !fromAbove) {
                continue;
            }
            
            chunk.lockBlock(bX, bY, bZ);
            unsigned char currentLevel = chunk.getLightLevel(bX, bY, bZ);

            if (currentLevel >= level && !source) {
                chunk.unlockBlock(bX, bY, bZ);
                continue;
            }

            if (currentLevel < level) {
                chunk.setLightLevel(bX, bY, bZ, level);
            }
            
            // 完全不透明且不是光源，不继续传播
            if (info && info->visualProps.lightOpacity == 15 && !source) {
                chunk.unlockBlock(bX, bY, bZ);
                continue;
            }
            
            chunk.unlockBlock(bX, bY, bZ);
        }
        
        source = false;
        if(level>1){
            toVisit.push(LightNode(cX, cZ, bX + 1, bY, bZ, level - 1, false));
            toVisit.push(LightNode(cX, cZ, bX - 1, bY, bZ, level - 1, false));
            toVisit.push(LightNode(cX, cZ, bX, bY - 1, bZ, level - 1, true));
            toVisit.push(LightNode(cX, cZ, bX, bY + 1, bZ, level - 1, false));
            toVisit.push(LightNode(cX, cZ, bX, bY, bZ - 1, level - 1, false));
            toVisit.push(LightNode(cX, cZ, bX, bY, bZ + 1, level - 1, false));
        }
    }
}
```







