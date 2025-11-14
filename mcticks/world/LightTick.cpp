#include "LightTick.h"
#include "Block.h"
#include "Chunk.h"
#include "ChunkCoord.h"
#include <cstdint>
#include <queue>

struct LightNode {
    int16_t cX, cZ;   // 区块坐标，范围小，用 int16 足够
    int16_t bX, bY, bZ; // 方块坐标：X/Z∈[0,15]，Y∈[-64,319]（都能放进 int16）
    uint8_t level;    // 光照等级 0..15
    uint8_t flags;    // bit0: fromAbove
    LightNode(int16_t cX,int16_t cZ,int16_t bX,int16_t bY,int16_t bZ,
            uint8_t level, uint8_t flags):cX(cX),cZ(cZ),bX(bX),bY(bY),bZ(bZ),level(level),flags(flags){}
    inline bool fromAbove() const { return flags & 1; }
};
static_assert(std::is_trivially_copyable<LightNode>::value, "POD required");

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

// 单线程版本的 lightTick 函数，使用二维数组
void lightTickArray(Chunk* chunksArray, int width, int height, int minX, int minZ, int realMinX, int realMinZ, int realMaxX, int realMaxZ) {
    // 计算有效区域的范围
    int realWidth = realMaxX - realMinX + 1;
    int realHeight = realMaxZ - realMinZ + 1;
    int totalRealChunks = realWidth * realHeight;
    
    // 步骤1：只清空有效区域的光照数据
    #pragma omp parallel for schedule(dynamic) num_threads(26)
    for (int idx = 0; idx < totalRealChunks; ++idx) {
        int offsetX = idx / realHeight;
        int offsetZ = idx % realHeight;
        int chunkX = realMinX + offsetX;
        int chunkZ = realMinZ + offsetZ;
        
        // 计算在完整数组中的索引
        int arrayIdx = (chunkX - minX) * height + (chunkZ - minZ);
        chunksArray[arrayIdx].clearLightData();
    }
    
    // 步骤2：只遍历有效区域的区块，对发光方块进行 Flood Fill
    #pragma omp parallel for schedule(dynamic) num_threads(26)
    for (int idx = 0; idx < totalRealChunks; ++idx) {
        int offsetX = idx / realHeight;
        int offsetZ = idx % realHeight;
        int chunkX = realMinX + offsetX;
        int chunkZ = realMinZ + offsetZ;
        
        // 计算在完整数组中的索引
        int arrayIdx = (chunkX - minX) * height + (chunkZ - minZ);
        Chunk& chunk = chunksArray[arrayIdx];
        
        for (int x = 0; x < 16; ++x) {
            for (int y = -64; y < 320; ++y) {
                for (int z = 0; z < 16; ++z) {
                    int block_type = chunk.getBlockID(x, y, z);
                    // 快速过滤：空气方块ID为0，直接跳过
                    if (block_type == 0) continue;
                    
                    const BlockInfo* info = globalBlockRegistry.getBlockInfo(block_type);
                    // 不是发光方块直接跳过
                    if (info->visualProps.lightEmission == 0) continue;
                    
                    floodFillLightArray(chunksArray, width, height, minX, minZ,
                                       realMinX, realMinZ, realMaxX, realMaxZ,
                                       chunkX, chunkZ,
                                       x, y, z,
                                       info->visualProps.lightEmission);
                }
            }
        }
    }
}
