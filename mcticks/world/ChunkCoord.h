#pragma once

// #include "BlockPos.h"
#include <cstddef>
#include "Tuple.h"

// Chunk 的坐标
struct ChunkCoord {
    int x;
    int z;
    
    ChunkCoord(int x = 0, int z = 0) : x(x), z(z) {}
    
    bool operator==(const ChunkCoord& other) const {
        return x == other.x && z == other.z;
    }
    
    struct Hash {
        size_t operator()(const ChunkCoord& coord) const {
            return coord.x ^ (coord.z << 16);
        }
    };
    
    static ChunkCoord fromBlockPos(const BlockPos& pos) {
        return {pos.x >> 4, pos.z >> 4};
    }
};