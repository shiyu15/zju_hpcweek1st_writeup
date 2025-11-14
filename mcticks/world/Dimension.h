#pragma once
#include "Chunk.h"
#include "LevelTicks.h"
#include "LightTick.h"
#include <string>
#include <random>
#include "BlockBehaviour.h"
#include "ChunkCoord.h"
#include <iostream>

class Dimension {
public:
    // 固定的世界范围：X 和 Z 都是 -8 到 31（包含）
    static constexpr int MIN_CHUNK_X = -8;
    static constexpr int MAX_CHUNK_X = 31;
    static constexpr int MIN_CHUNK_Z = -8;
    static constexpr int MAX_CHUNK_Z = 31;
    static constexpr int CHUNK_WIDTH = MAX_CHUNK_X - MIN_CHUNK_X + 1;   // 40
    static constexpr int CHUNK_HEIGHT = MAX_CHUNK_Z - MIN_CHUNK_Z + 1;  // 40
    static constexpr int TOTAL_CHUNKS = CHUNK_WIDTH * CHUNK_HEIGHT;     // 1600

    int REAL_MIN_CHUNK_X = 1E5;
    int REAL_MAX_CHUNK_X = -1E5;
    int REAL_MIN_CHUNK_Z = 1E5;
    int REAL_MAX_CHUNK_Z = -1E5;
    int _flag;

    Dimension(const std::string &name) : blockTicks(LevelTicks()), random(std::random_device()()) {
        // 初始化固定大小的二维数组
        chunksArray = new Chunk[TOTAL_CHUNKS];
        _flag = 0;
    }
    
    ~Dimension() {
        delete[] chunksArray;
    }
    
    // 获取区块（二维数组索引）
    inline Chunk* getChunk(int chunkX, int chunkZ) {
        if (chunkX < MIN_CHUNK_X || chunkX > MAX_CHUNK_X) return nullptr;
        if (chunkZ < MIN_CHUNK_Z || chunkZ > MAX_CHUNK_Z) return nullptr;
        int idx = (chunkX - MIN_CHUNK_X) * CHUNK_HEIGHT + (chunkZ - MIN_CHUNK_Z);
        return &chunksArray[idx];
    }
    
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
    
    // 处理所有计划刻
    void processTicks() {
        blockTicks.tick([this](BlockPos pos, LevelTicks &ticks) { this->tickBlock(pos, ticks);},
                        [this](BlockPos pos, int blockID) { this->tickSetBlockID(pos, blockID);});
    }

    // 统一执行 GO 发起的 SetBlock 得到的计划刻
    void setBlockTick() {
        blockTicks.setBlockTick([this](BlockPos pos, LevelTicks &ticks) { this->tickBlock(pos, ticks);},
                        [this](BlockPos pos, int blockID) { this->tickSetBlockID(pos, blockID);});
    }
    
    // 在所有计划刻之前的预处理，即设置方块 ID，不让方块的顺序影响行为
    void tickSetBlockID(const BlockPos& pos, int blockID) {
        setBlockID(pos, blockID);
    }

    // 执行方块的行为逻辑
    void tickBlock(BlockPos pos, LevelTicks &ticks) {
        int blockType = getBlockID(pos);
        BlockBehaviour* behaviour = globalBlockStateRegistry.getBlockStateBehaviour(blockType);
        behaviour->tick(this, pos, ticks);
        behaviour->randomTick(this, pos, random, ticks);
    }

    // 获取指定位置的方块 ID
    int getBlockID(const BlockPos& pos) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        
        if (chunksArray != nullptr) {
            Chunk* chunk = getChunk(chunkCoord.x, chunkCoord.z);
            if (chunk == nullptr) return 0;
            int relX = pos.x & 0xF;
            int relY = pos.y;
            int relZ = pos.z & 0xF;
            return chunk->getBlockID(relX, relY, relZ);
        }
        
    }

    // 设置指定位置的方块 ID
    void setBlockID(const BlockPos& pos, int blockID) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        
        if (chunksArray != nullptr) {
            Chunk* chunk = getChunk(chunkCoord.x, chunkCoord.z);
            if (chunk == nullptr) return;
            int relX = pos.x & 0xF;
            int relY = pos.y;
            int relZ = pos.z & 0xF;
            chunk->setBlockID(relX, relY, relZ, blockID);
            return;
        }
    }

    LevelTicks blockTicks;
    
    // 二维数组存储（用于优化访问）
    Chunk* chunksArray;
    int minChunkX, minChunkZ;
    int chunkWidth, chunkHeight;
    
private:
    int chunkCount = 0;
    std::string name;
    std::mt19937 random;
};
