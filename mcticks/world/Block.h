#pragma once
#include <string>
#include <vector>
#include "BlockInfo.h"
#include <iostream>

class BlockRegistry {
public:
    // 最大方块ID（根据实际使用的最大ID设置）
    static constexpr int MAX_BLOCK_ID = 20000;
    
    BlockRegistry() {
        // 预分配数组，初始化为nullptr表示未注册
        registry.resize(MAX_BLOCK_ID, nullptr);
        
        // 初始化默认方块信息
        defaultBlockInfo = new BlockInfo(
            "Default Block",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        );
    }
    
    ~BlockRegistry() {
        // 清理动态分配的内存
        for (auto* info : registry) {
            if (info != nullptr && info != defaultBlockInfo) {
                delete info;
            }
        }
        delete defaultBlockInfo;
    }
    
    // 注册方块信息，stage 表示有多少个状态，比如水有 16 个状态
    void registerBlock(int id, const BlockInfo& info, int stage = 1) {
        for (int i = 0; i < stage; ++i) {
            int blockId = id + i;
            if (blockId >= MAX_BLOCK_ID) continue;
            
            BlockInfo* newInfo = new BlockInfo(info);
            newInfo->stage = i % 8;
            registry[blockId] = newInfo;
        }
    }

    // 获取方块信息 - O(1) 时间复杂度，无除法运算
    inline const BlockInfo* getBlockInfo(int id) const {
        if (id < 0 || id >= MAX_BLOCK_ID) {
            return defaultBlockInfo;
        }
        const BlockInfo* info = registry[id];
        return info != nullptr ? info : defaultBlockInfo;
    }
    
private:
    std::vector<const BlockInfo*> registry;
    BlockInfo* defaultBlockInfo;
};

extern BlockRegistry globalBlockRegistry;



