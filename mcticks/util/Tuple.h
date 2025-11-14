#pragma once
#include <cstddef>
#include <functional>
struct Tuple9{
    int a0, a1, a2;
    int a3, a4, a5;
    int a6, a7, a8;
    Tuple9(int a0=0, int a1=0, int a2=0,
           int a3=0, int a4=0, int a5=0,
           int a6=0, int a7=0, int a8=0)
        : a0(a0), a1(a1), a2(a2),
          a3(a3), a4(a4), a5(a5),
          a6(a6), a7(a7), a8(a8) {}
};

struct Tuple3 {
    float x;
    float y;
    float z;
    Tuple3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
};

struct BlockPos {
    int x;
    int y;
    int z;

    BlockPos() : x(0), y(0), z(0) {}
    
    BlockPos(int x, int y, int z) : x(x), y(y), z(z) {}
    
    bool operator==(const BlockPos& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const BlockPos& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    struct Hash {
        size_t operator()(const BlockPos& pos) const {
            return pos.y ^ (pos.x << 10) ^ (pos.z << 24);
        }
    };
};