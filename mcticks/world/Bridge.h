#ifndef BRIDGE_H
#define BRIDGE_H

#include <cstdint>
#include <iostream>
#include <stdint.h>
#include <unordered_map>
#include "LevelTicks.h"
#include "Chunk.h"
#include "Dimension.h"
#include "Tuple.h"


extern "C" {
struct SetblockRequest{
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t state_id;
};

void clear_ticks();
BrChunk* load_chunk(int32_t x, int32_t z);
void tick_chunk();
void setblock(int32_t x, int32_t y, int32_t z, int32_t state_id);
void batch_setblock(size_t len, struct SetblockRequest* reqs);
void tickAftersetblock();
}


class Bridging {
private:
    Dimension* dimension;
    friend BrChunk* load_chunk(int32_t x, int32_t z);
    std::unordered_map<BlockPos, int, BlockPos::Hash> setBlockBuffer;

public:
    Bridging(std::string dimension_name) : dimension(new Dimension(dimension_name)) {
        setBlockBuffer.clear();
    }

    ~Bridging() { delete dimension; }

    // 清除所有计划刻
    void clearTicks() {
        dimension->blockTicks.clearTicks();
    }

    // 执行维度的 tick
    void tickDimension() { 
        dimension->tick();
    }

    // 告诉 GO 端所有区块都更新了，需要发往客户端
    void tick() {
        // 更新所有区块的 last_update
        for (int i = 0; i < Dimension::TOTAL_CHUNKS; ++i) {
            dimension->chunksArray[i].brchunk.last_update++;
        }
    }

    // 根据区块 id 获取区块数据
    BrChunk* getBrChunk(ChunkCoord coord) {
        auto it = dimension->getChunk(coord);
        return &it->brchunk;
    }

    // 统一更新 GO 发起的 SetBlock 操作导致的 BlockID 变动（包括触发的 Tick）
    void setBlockTick()  {
        // for (const auto& [pos, state_id] : setBlockBuffer) {
        //     dimension->setBlockID(pos, state_id);
        // }
        setBlockBuffer.clear();
        dimension->setBlockTick();
    }

    // 设置方块，注意这里并不直接修改方块，而是放入一个缓冲区，等到 setBlockTick 的时候再统一修改
    void setblock(int32_t x, int32_t y, int32_t z, int32_t state_id) {
        BlockPos pos(x, y, z);
        // setBlockBuffer[pos] = state_id;
        dimension->setBlockID(pos, state_id);
        dimension->blockTicks.addTick(-1, pos, 0); // 立刻显示在玩家面前
        dimension->blockTicks.addTick(-1, pos, 1); // 人为的操作是对下一个 tick 强制有效的，优先级最高。
        int dx[6] = {1, -1, 0, 0, 0, 0};
        int dy[6] = {0, 0, 1, -1, 0, 0};
        int dz[6] = {0, 0, 0, 0, 1, -1};
        for (int dir = 0; dir < 6; dir++) {
            BlockPos neighbor(x + dx[dir], y + dy[dir], z + dz[dir]);
            int ntype = dimension->getBlockID(neighbor);
            dimension->blockTicks.addTick(ntype, neighbor, 0);
            dimension->blockTicks.addTick(ntype, neighbor, 1);
        }
    }
};

#endif // BRIDGE_H