#pragma once
#include "Tuple.h"
#include <random>
#include <vector>
#include <unordered_set>
#include "LevelTicks.h"
class Dimension;

// 方块行为接口，赛题目前实现了水和岩浆的行为
class BlockBehaviour {
public:
    BlockBehaviour() {}
    
    virtual ~BlockBehaviour() = default;
    
    virtual void tick(Dimension* dimension, const BlockPos& pos, LevelTicks &ticks) {

    }
    
    virtual void randomTick(Dimension* dimension, const BlockPos& pos, std::mt19937& random, LevelTicks &ticks) {

    }
};


// 根据方块类型映射对应的方块行为。
class BlockStateRegistry {
public:
    // 最大方块状态ID（与BlockRegistry保持一致）
    static constexpr int MAX_BLOCK_STATE_ID = 20000;
    
    BlockStateRegistry() {
        // 预分配数组，初始化为默认行为
        defaultBehaviour = new BlockBehaviour();
        registry_.resize(MAX_BLOCK_STATE_ID, defaultBehaviour);
    }
    
    ~BlockStateRegistry() {
        // 清理动态分配的行为对象（避免重复删除defaultBehaviour）
        std::unordered_set<BlockBehaviour*> uniqueBehaviours;
        for (auto* behaviour : registry_) {
            if (behaviour != nullptr && behaviour != defaultBehaviour) {
                uniqueBehaviours.insert(behaviour);
            }
        }
        for (auto* behaviour : uniqueBehaviours) {
            delete behaviour;
        }
        delete defaultBehaviour;
    }

    void registerBlockState(int blockType, BlockBehaviour* behaviour) {
        if (blockType >= 0 && blockType < MAX_BLOCK_STATE_ID) {
            registry_[blockType] = behaviour;
        }
    }
    
    // 获取方块行为 - O(1) 时间复杂度，无除法运算
    inline BlockBehaviour* getBlockStateBehaviour(int blockType) {
        if (blockType < 0 || blockType >= MAX_BLOCK_STATE_ID) {
            return defaultBehaviour;
        }
        return registry_[blockType];
    }
    
private:
    std::vector<BlockBehaviour*> registry_;
    BlockBehaviour* defaultBehaviour; // 默认返回空气方块状态
};

extern BlockStateRegistry globalBlockStateRegistry;