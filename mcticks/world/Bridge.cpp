// 和 GO 的交互，不允许修改
#include "Bridge.h"

Bridging *globalBridging;

extern "C" __attribute__((visibility("default")))
  BrChunk* load_chunk(int32_t x, int32_t z){
    ChunkCoord coord(x, z);
    return globalBridging->getBrChunk(coord);
};

extern "C" __attribute__((visibility("default")))
void tick_chunk(){
    globalBridging->tickDimension();
    globalBridging->tick();   
}

extern "C" __attribute__((visibility("default")))
void setblock(int32_t x, int32_t y, int32_t z, int32_t state_id){
    globalBridging->setblock(x, y, z, state_id);
    globalBridging->tick(); 
}

extern "C" __attribute__((visibility("default")))
void batch_setblock(size_t len, struct SetblockRequest* reqs){
    for (size_t i = 0; i < len; i++) {
        const SetblockRequest& req = reqs[i];
        globalBridging->setblock(req.x, req.y, req.z, req.state_id);
    }
    globalBridging->tick();
}

extern "C" __attribute__((visibility("default")))
void clear_ticks(){
    globalBridging->clearTicks();
    globalBridging->tick();
}

extern "C" __attribute__((visibility("default")))
void tickAftersetblock(){
    globalBridging->setBlockTick();
    globalBridging->tick();
}
__attribute__((constructor))
void init_bridge() {
    std::cerr << "Bridge initialized! " << std::endl;
    globalBridging = new Bridging("Overworld");
}
