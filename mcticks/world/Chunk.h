#pragma once
#include <cstring>
#include <cstdint>
#include <atomic>
extern int MAX_CHUNKS;


// Chunk 的 NBT 结构定义
extern "C"{
typedef struct {
  int16_t blockcount;
  int32_t blocks_state[4096];
  int32_t biomes[64];

  uint8_t  sky_light[2048];
  uint8_t  block_light[2048];
} Section;
typedef struct {
    int32_t last_update;
    Section sections[24];
} BrChunk; 
}

class Chunk {
public:

    Chunk() {
        std::memset(&brchunk, 0, sizeof(BrChunk));
        // set default sky light to max
        for (int i = 0; i < 24; ++i) {
            std::memset(brchunk.sections[i].sky_light, 0xFF, sizeof(brchunk.sections[i].sky_light));
        }
        // 初始化自旋锁
        for (int i = 0; i < 24; ++i) {
            for (int j = 0; j < 2048; ++j) {
                blockSpinLocks[i][j].clear(std::memory_order_release);
            }
        }
    }

    ~Chunk() {
    }

    inline int getBlockID(int x, int y, int z) {
        if (y < -64 || y >= 320) {
            return 0;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        return brchunk.sections[sectionY].blocks_state[localIndex];
    }

    inline void setBlockID(int x, int y, int z, int state_id) {
        if (y < -64 || y >= 320) {
            return ;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        brchunk.sections[sectionY].blocks_state[localIndex] = state_id;
    }

    inline unsigned char getLightLevel(int x, int y, int z) const {
        if (y < -64 || y >= 320) {
            return 0;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        unsigned char currentByte = brchunk.sections[sectionY].block_light[localIndex / 2] ;
        return (localIndex % 2 == 0) ? (currentByte & 0x0F) : (currentByte >> 4);
    }

    inline void setLightLevel(int x, int y, int z, unsigned char level) {
        if (y < -64 || y >= 320) {
            return ;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        int byteIndex = localIndex / 2;
        unsigned char &currentByte = brchunk.sections[sectionY].block_light[byteIndex];
        if (localIndex % 2 == 0) {
            currentByte = (currentByte & 0xF0) | (level & 0x0F);
        } else {
            currentByte = (currentByte & 0x0F) | ((level & 0x0F) << 4);
        }
    }
    
    // 获取指定block的自旋锁
    inline std::atomic_flag& getBlockSpinLock(int x, int y, int z) {
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        int byteIdx  = localIndex / 2;               // 关键：按字节加锁
        return blockSpinLocks[sectionY][byteIdx];
    }
    
    // 自旋锁的lock操作
    inline void lockBlock(int x, int y, int z) {
        while (getBlockSpinLock(x, y, z).test_and_set(std::memory_order_acquire)) {
            // 自旋等待
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();  // x86的pause指令，减少CPU功耗
            #endif
        }
    }
    
    // 自旋锁的unlock操作
    inline void unlockBlock(int x, int y, int z) {
        getBlockSpinLock(x, y, z).clear(std::memory_order_release);
    }

    void clearLightData() {
        for (int i = 0; i < 24; ++i) {
            std::memset(brchunk.sections[i].block_light, 0, sizeof(brchunk.sections[i].block_light));
        }
    }
    
    BrChunk brchunk;

    // 每个section的block自旋锁数组 (24个section，每个section有2048个字节)
    std::atomic_flag blockSpinLocks[24][2048];
private:
    
    int getSectionY(int y) const {
        return (y + 64) / 16;
    }

    int getIndex(int x, int y, int z) const {
        int localY = y + 64 - getSectionY(y) * 16;
        return (localY * 16 * 16) + (z * 16) + x;
    }

    friend class Bridging;

};