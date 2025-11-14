#include "BlockBehaviour.h"
#include "Dimension.h"
#include "LevelTicks.h"
#include "Block.h"
#include "Tuple.h"
#include <iostream>
#include <set>
#define WATER_TICK_RATE 1
#define LAVA_TICK_RATE 1
BlockStateRegistry globalBlockStateRegistry;

// 水和岩浆方块的行为逻辑，其他方块的行为为空
class WaterBlockBehaviour : public BlockBehaviour {
public:
    WaterBlockBehaviour() {}

    void randomTick(Dimension* dimension, const BlockPos& pos, std::mt19937& random, LevelTicks &ticks) {
    }
    
    void tick(Dimension* dimension, const BlockPos& pos, LevelTicks &ticks) {
        int water_type = dimension->getBlockID(pos);
        // 不是水方块，走它本身对应的行为
        if (water_type < SourceID || water_type > SourceID + 15) {
            globalBlockStateRegistry.getBlockStateBehaviour(water_type)->tick(dimension, pos, ticks);
            return; 
        }
        
        // 6 个方向
        BlockPos up_pos(pos.x, pos.y + 1, pos.z);
        BlockPos down_pos(pos.x, pos.y - 1, pos.z);
        BlockPos left_pos(pos.x - 1, pos.y, pos.z);
        BlockPos right_pos(pos.x + 1, pos.y, pos.z);
        BlockPos front_pos(pos.x, pos.y, pos.z - 1);
        BlockPos back_pos(pos.x, pos.y, pos.z + 1);

        int up_block_type = dimension->getBlockID(up_pos);
        int down_block_type = dimension->getBlockID(down_pos);
        int left_block_type = dimension->getBlockID(left_pos);
        int right_block_type = dimension->getBlockID(right_pos);
        int front_block_type = dimension->getBlockID(front_pos);
        int back_block_type = dimension->getBlockID(back_pos);

        // 无限水逻辑
        if (!isSource(water_type)) {
            // 如果下方是水源或者是固体方块，并且旁边至少有两个水源方块，则变成水源
            if ((isSource(down_block_type)) || isSolid(down_block_type)) {
                int num = isSource(left_block_type) + isSource(right_block_type) + isSource(front_block_type) + isSource(back_block_type);
                if (num >= 2) {
                    // 在下一个 tick 变成水源
                    ticks.addTick(SourceID, pos, WATER_TICK_RATE);
                    return;
                }
            }
        }
    
        
        // 向下流动的逻辑，如果下方是空气或者是非水源的水流，则向下流动
        if (down_block_type == 0 || (isWater(down_block_type) && !isSource(down_block_type))) {
            // sourceID + 8 是 MC 定义的下落的水流
            int down_type = SourceID + 8;
            ticks.addTick(down_type, down_pos, WATER_TICK_RATE);
            // 流水消失逻辑
            if (!isSource(water_type)) {
                // 如果上方没有水流，并且旁边没有比它等级更低的水流（等级 0 是水源），则消失
                if (!isWater(up_block_type) && isFather(left_block_type, water_type) + isFather(right_block_type, water_type) +
                    isFather(front_block_type, water_type) + isFather(back_block_type, water_type) == 0) {
                    // 负数表示是为了强调其优先级，消失的水流优先级高于普通水流
                    if (water_type >= SourceID + 7) {
                        // 如果是最低等级的水流，直接消失，这里使用 - (SourceID + 8) 来表示空气（特判）
                        ticks.addTick(-(SourceID + 8), pos, WATER_TICK_RATE);
                    } else {
                        // 如果是其他等级的水流，降低一个等级
                        ticks.addTick(-(water_type + 1), pos, WATER_TICK_RATE);
                    }
                    ticks.addTick(left_block_type, left_pos, WATER_TICK_RATE);
                    ticks.addTick(right_block_type, right_pos, WATER_TICK_RATE);
                    ticks.addTick(front_block_type, front_pos, WATER_TICK_RATE);
                    ticks.addTick(back_block_type, back_pos, WATER_TICK_RATE);
                    return;
                }
            }
            return;
        } else {
            // tick 正下方的方块，下方是岩浆的的特殊情况
            ticks.addTick(down_block_type, down_pos, WATER_TICK_RATE);
        }

        // 水平流动的逻辑
        spreadWaterHorizontally(dimension, pos, ticks);

        // 流水消失逻辑
        if (!isSource(water_type)) {
            // 如果上方没有水流，并且旁边没有比它等级更低的水流（等级 0 是水源），则消失
            if (!isWater(up_block_type) && isFather(left_block_type, water_type) + isFather(right_block_type, water_type) +
                isFather(front_block_type, water_type) + isFather(back_block_type, water_type) == 0) {
                // 负数表示是为了强调其优先级，消失的水流优先级高于普通水流
                if (water_type >= SourceID + 7) {
                    // 如果是最低等级的水流，直接消失，这里使用 - (SourceID + 8) 来表示空气（特判）
                    ticks.addTick(-(SourceID + 8), pos, WATER_TICK_RATE);
                } else {
                    // 如果是其他等级的水流，降低一个等级
                    ticks.addTick(-(water_type + 1), pos, WATER_TICK_RATE);
                }
                // 同时 tick 它旁边的方块，为了提醒周围的水流它消失了，可能会影响到它们
                ticks.addTick(down_block_type, down_pos, WATER_TICK_RATE);
                ticks.addTick(left_block_type, left_pos, WATER_TICK_RATE);
                ticks.addTick(right_block_type, right_pos, WATER_TICK_RATE);
                ticks.addTick(front_block_type, front_pos, WATER_TICK_RATE);
                ticks.addTick(back_block_type, back_pos, WATER_TICK_RATE);
                return;
            }
        }
    }
    
private:
    static constexpr int dx[4] = {0, 1, 0, -1};
    static constexpr int dz[4] = {-1, 0, 1, 0};
    static const int SourceID = 86; 

    inline bool isSource(const int blockID) {
        return blockID == SourceID;
    };

    inline bool isWater(const int blockID) {
        return blockID >= SourceID && blockID <= SourceID + 15;
    };

    inline bool isLava(const int blockID) {
        return blockID >= 102 && blockID <= 102 + 15;
    };

    // 判断 Block 和 compare 之间的水流等级
    inline bool isFather(const int blockID, const int compareID) {
        if (!isWater(blockID)) return false;
        if (isWater(blockID) && !isWater(compareID)) return true;
        return globalBlockRegistry.getBlockInfo(blockID)->stage < globalBlockRegistry.getBlockInfo(compareID)->stage;
    };
        

    void spreadWaterHorizontally(Dimension* dimension, const BlockPos& pos, LevelTicks& ticks) {
        int water_type = dimension->getBlockID(pos);
        int level = globalBlockRegistry.getBlockInfo(water_type)->stage;

        int weights[4] = {999, 999, 999, 999};

        for (int dir = 0; dir < 4; dir++) {
            BlockPos neighbor(pos.x + dx[dir], pos.y, pos.z + dz[dir]);
            int neighbor_block_type = dimension->getBlockID(neighbor);
            if (isLava(neighbor_block_type)) {
                ticks.addTick(neighbor_block_type, neighbor, LAVA_TICK_RATE);
                continue;
            }
            int neighbor_type = neighbor_block_type;
            // 如果邻居是固体或者比它最多高 1 级（说明已经tick过了）的水流，则不能流过去
            if (!isSolid(neighbor_block_type) && !(isWater(neighbor_block_type) && globalBlockRegistry.getBlockInfo(neighbor_type)->stage <= level + 1)) {
                // 水往低处流，寻找 4 格之内能下降的最少的路径
                weights[dir] = findShortestPath(dimension, neighbor, dir, 4);
            } else {
                weights[dir] = 1000;
            }
        }

        int minWeight = 999;
        for (int dir = 0; dir < 4; dir++) {
            if (weights[dir] < minWeight) {
                minWeight = weights[dir];
            }
        }

        if (level < 7) {
            // 流向最优的方向
            for (int dir = 0; dir < 4; dir++) {
                if (weights[dir] == minWeight) {
                    BlockPos target(pos.x + dx[dir], pos.y, pos.z + dz[dir]);
                    // 流向这个方向，水流等级 + 1
                    ticks.addTick(SourceID + 1 + level, target, WATER_TICK_RATE);
                    BlockPos down_target(target.x, target.y - 1, target.z);
                    // tick 邻居正下方的方块，下方是岩浆的的特殊情况
                    ticks.addTick(dimension->getBlockID(down_target), down_target, WATER_TICK_RATE);
                }
            }
        }
    }

    // 寻找低一格的最近的非实体方块，这个方向就是
    int findShortestPath(Dimension* dimension, const BlockPos& start, int dir, int maxDepth) {
        // BFS 算法
        struct Node { BlockPos pos; int dist; };
        std::queue<Node> q;
        std::set<BlockPos> visited;

        q.push({start, 0});
        visited.insert(start);

        int bestDist = 999;

        while (!q.empty()) {
            Node cur = q.front();
            q.pop();

            if (cur.dist > maxDepth) continue;

            int block_type = dimension->getBlockID(cur.pos);

            BlockPos below(cur.pos.x, cur.pos.y - 1, cur.pos.z);
            if (visited.count(below) == 0) {
                int below_block_type = dimension->getBlockID(below);
                // 找到了一个能允许下降的位置
                if (!isSolid(below_block_type)) {
                    bestDist = std::min(bestDist, cur.dist);
                    continue;
                }
            }

            // 向四个方向扩展查找
            for (int i = 0; i < 4; i++) {
                BlockPos next(cur.pos.x + dx[i], cur.pos.y, cur.pos.z + dz[i]);
                if (visited.count(next)) continue;
                int next_block_type = dimension->getBlockID(next);
                if (!isSolid(next_block_type)) {
                    q.push({next, cur.dist + 1});
                    visited.insert(next);
                }
            }
        }

        return bestDist;
    }

    
    bool isSolid(const int blockID) {
        return globalBlockRegistry.getBlockInfo(blockID)->interactionProps.canWalkThrough == false;
    }
    
};

class LavaBlockBehaviour : public BlockBehaviour {
public:
    LavaBlockBehaviour() {}

    void randomTick(Dimension* dimension, const BlockPos& pos, std::mt19937& random, LevelTicks &ticks) {
    }
    
    // 岩浆的行为逻辑和水类似，但是需要一些特殊判断
    void tick(Dimension* dimension, const BlockPos& pos, LevelTicks &ticks) {
        int lava_type = dimension->getBlockID(pos);
        if (lava_type < SourceID || lava_type > SourceID + 15) {
            globalBlockStateRegistry.getBlockStateBehaviour(lava_type)->tick(dimension, pos, ticks);
            return; 
        }
        

        BlockPos up_pos(pos.x, pos.y + 1, pos.z);
        BlockPos down_pos(pos.x, pos.y - 1, pos.z);
        BlockPos left_pos(pos.x - 1, pos.y, pos.z);
        BlockPos right_pos(pos.x + 1, pos.y, pos.z);
        BlockPos front_pos(pos.x, pos.y, pos.z - 1);
        BlockPos back_pos(pos.x, pos.y, pos.z + 1);

        int up_block_type = dimension->getBlockID(up_pos);
        int down_block_type = dimension->getBlockID(down_pos);
        int left_block_type = dimension->getBlockID(left_pos);
        int right_block_type = dimension->getBlockID(right_pos);
        int front_block_type = dimension->getBlockID(front_pos);
        int back_block_type = dimension->getBlockID(back_pos);
        
        int flag = 0;

        
        // 向下流动的逻辑
        if (down_block_type == 0 || (isLava(down_block_type) && !isSource(down_block_type))) {
            int down_type = SourceID + 8;
            ticks.addTick(down_type, down_pos, LAVA_TICK_RATE);
            if (!isSource(lava_type)) {
                if (!isLava(up_block_type) && isFather(left_block_type, lava_type) + isFather(right_block_type, lava_type) +
                    isFather(front_block_type, lava_type) + isFather(back_block_type, lava_type) == 0) {
                    if (lava_type >= SourceID + 7) {
                        ticks.addTick(-(SourceID + 8), pos, LAVA_TICK_RATE);
                    } else {
                        ticks.addTick(-(lava_type + 1), pos, LAVA_TICK_RATE);
                    }
                    ticks.addTick(left_block_type, left_pos, LAVA_TICK_RATE);
                    ticks.addTick(right_block_type, right_pos, LAVA_TICK_RATE);
                    ticks.addTick(front_block_type, front_pos, LAVA_TICK_RATE);
                    ticks.addTick(back_block_type, back_pos, LAVA_TICK_RATE);
                }
            }
            return;
        }

        spreadLavaHorizontally(dimension, pos, ticks);

        // 岩浆消失逻辑，不一样的是消失的时候可能会和水反应生成石头
        if (!isSource(lava_type)) {
            if (!isLava(up_block_type) && isFather(left_block_type, lava_type) + isFather(right_block_type, lava_type) +
                isFather(front_block_type, lava_type) + isFather(back_block_type, lava_type) == 0) {
                flag = 1;
                ticks.addTick(down_block_type, down_pos, LAVA_TICK_RATE);
                ticks.addTick(left_block_type, left_pos, LAVA_TICK_RATE);
                ticks.addTick(right_block_type, right_pos, LAVA_TICK_RATE);
                ticks.addTick(front_block_type, front_pos, LAVA_TICK_RATE);
                ticks.addTick(back_block_type, back_pos, LAVA_TICK_RATE);
            }
        }

        if (isWater(down_block_type)) {
            if (isSource(lava_type)) {
                // 生成黑曜石
                ticks.addTick(2397, pos, LAVA_TICK_RATE);
            } else {
                // 下面是水，生成石头
                ticks.addTick(1, pos, LAVA_TICK_RATE);
            }
            flag = 0;
            ticks.addTick(up_block_type, up_pos, LAVA_TICK_RATE);
            ticks.addTick(down_block_type, down_pos, LAVA_TICK_RATE);
            ticks.addTick(left_block_type, left_pos, LAVA_TICK_RATE);
            ticks.addTick(right_block_type, right_pos, LAVA_TICK_RATE);
            ticks.addTick(front_block_type, front_pos, LAVA_TICK_RATE);
            ticks.addTick(back_block_type, back_pos, LAVA_TICK_RATE);
            return;
        }


        if (isWater(up_block_type) || isWater(left_block_type) || isWater(right_block_type) || isWater(front_block_type) || isWater(back_block_type)) {
            if (isSource(lava_type)) {
                // 生成黑曜石
                ticks.addTick(2397, pos, LAVA_TICK_RATE);
            } else {
                // 四周或者下方是水，生成圆石
                ticks.addTick(14, pos, LAVA_TICK_RATE);
            }
            flag = 0;
            ticks.addTick(up_block_type, up_pos, LAVA_TICK_RATE);
            ticks.addTick(down_block_type, down_pos, LAVA_TICK_RATE);
            ticks.addTick(left_block_type, left_pos, LAVA_TICK_RATE);
            ticks.addTick(right_block_type, right_pos, LAVA_TICK_RATE);
            ticks.addTick(front_block_type, front_pos, LAVA_TICK_RATE);
            ticks.addTick(back_block_type, back_pos, LAVA_TICK_RATE);
            return;
        }
        // 岩浆消失了
        if (flag) {
            if (lava_type >= SourceID + 7) {
                ticks.addTick(-(SourceID + 8), pos, LAVA_TICK_RATE);
            } else {
                ticks.addTick(-(lava_type + 1), pos, LAVA_TICK_RATE);
            }
            return;
        } 

    }
    
private:
    static constexpr int dx[4] = {0, 1, 0, -1};
    static constexpr int dz[4] = {-1, 0, 1, 0};
    static const int SourceID = 102; 
    static const int WaterID = 86;

    bool isWater(const int blockID) {
        return blockID >= WaterID && blockID <= WaterID + 15;
    }

    bool isSource(const int blockID) {
        return blockID == SourceID;
    }

    bool isLava(const int blockID) {
        return blockID >= SourceID && blockID <= SourceID + 15;
    }

    bool isFather(const int blockID, const int compareID) {
        if (!isLava(blockID)) return false;
        if (isLava(blockID) && !isLava(compareID)) return true;
        return globalBlockRegistry.getBlockInfo(blockID)->stage < globalBlockRegistry.getBlockInfo(compareID)->stage;
    }


    void spreadLavaHorizontally(Dimension* dimension, const BlockPos& pos, LevelTicks& ticks) {
        int lava_type = dimension->getBlockID(pos);
        int level = globalBlockRegistry.getBlockInfo(lava_type)->stage;

        int weights[4] = {999, 999, 999, 999};

        for (int dir = 0; dir < 4; dir++) {
            BlockPos neighbor(pos.x + dx[dir], pos.y, pos.z + dz[dir]);
            int neighbor_block_type = dimension->getBlockID(neighbor);

            if (!isSolid(neighbor_block_type) && !(neighbor_block_type >= SourceID && neighbor_block_type <= SourceID + 15 && globalBlockRegistry.getBlockInfo(neighbor_block_type)->stage <= level + 1)) {
                weights[dir] = findShortestPath(dimension, neighbor, dir, 4);
            } else {
                weights[dir] = 1000;
            }
        }

        int minWeight = 999;
        for (int dir = 0; dir < 4; dir++) {
            if (weights[dir] < minWeight) {
                minWeight = weights[dir];
            }
        }

        if (level < 7) {
            for (int dir = 0; dir < 4; dir++) {
                if (weights[dir] == minWeight) {
                    BlockPos target(pos.x + dx[dir], pos.y, pos.z + dz[dir]);
                    ticks.addTick(SourceID + 1 + level, target, LAVA_TICK_RATE);
                }
            }
        }
    }

    int findShortestPath(Dimension* dimension, const BlockPos& start, int dir, int maxDepth) {
        struct Node { BlockPos pos; int dist; };
        std::queue<Node> q;
        std::set<BlockPos> visited;

        q.push({start, 0});
        visited.insert(start);

        int bestDist = 999;

        while (!q.empty()) {
            Node cur = q.front();
            q.pop();

            if (cur.dist > maxDepth) continue;

            int block_type = dimension->getBlockID(cur.pos);

            BlockPos below(cur.pos.x, cur.pos.y - 1, cur.pos.z);
            if (visited.count(below) == 0) {
                int below_block_type = dimension->getBlockID(below);
                if (!isSolid(below_block_type)) {
                    bestDist = std::min(bestDist, cur.dist);
                    continue;
                }
            }

            for (int i = 0; i < 4; i++) {
                BlockPos next(cur.pos.x + dx[i], cur.pos.y, cur.pos.z + dz[i]);
                if (visited.count(next)) continue;
                int next_block_type = dimension->getBlockID(next);

                if (!isSolid(next_block_type)) {
                    q.push({next, cur.dist + 1});
                    visited.insert(next);
                }
            }
        }

        return bestDist;
    }

    
    bool isSolid(const int blockID) {
        return globalBlockRegistry.getBlockInfo(blockID)->interactionProps.canWalkThrough == false;
    }
    
};


struct BlockStateInitializer {
    BlockStateInitializer() {
        globalBlockStateRegistry.registerBlockState(86, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(87, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(88, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(89, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(90, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(91, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(92, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(93, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(94, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(95, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(96, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(97, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(98, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(99, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(100, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(101, new WaterBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(102, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(103, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(104, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(105, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(106, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(107, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(108, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(109, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(110, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(111, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(112, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(113, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(114, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(115, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(116, new LavaBlockBehaviour());
        globalBlockStateRegistry.registerBlockState(117, new LavaBlockBehaviour());
    }
} blockStateIntializer;