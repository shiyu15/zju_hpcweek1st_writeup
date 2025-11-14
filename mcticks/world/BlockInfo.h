#pragma once
#include <string>


// 方块信息，包括名称、视觉属性和交互属性
class BlockInfo {
public:
     
    struct VisualProperties {
        int lightEmission;  // 0-15 光照等级
        int lightOpacity;   // 0: fully transparent, 15: opaque

        VisualProperties() = default;
        VisualProperties(int le, int lo)
            : lightEmission(le), lightOpacity(lo) {}
    };

    struct InteractionProperties {
        bool canPlaceOn;        // 是否可以放置在上面
        bool canWalkThrough;    // 是否可以穿过
        InteractionProperties() = default;
        InteractionProperties(bool cpo, bool cwt)
            : canPlaceOn(cpo), canWalkThrough(cwt) {}
    };

    std::string name;
    int stage;
    VisualProperties visualProps;
    InteractionProperties interactionProps;
    BlockInfo() = default;
    BlockInfo(std::string n, const VisualProperties& vp,
            const InteractionProperties& ip)
        : name(std::move(n)), stage(0), visualProps(vp.lightEmission, vp.lightOpacity),
        interactionProps(ip.canPlaceOn, ip.canWalkThrough) {}

    BlockInfo(const BlockInfo& other) {
        name = other.name;
        stage = other.stage;
        visualProps.lightEmission = other.visualProps.lightEmission;
        visualProps.lightOpacity = other.visualProps.lightOpacity;
        interactionProps.canPlaceOn = other.interactionProps.canPlaceOn;
        interactionProps.canWalkThrough = other.interactionProps.canWalkThrough;
    }

    BlockInfo& operator=(const BlockInfo& other) {
        if (this != &other) {
            name = other.name;
            stage = other.stage;
            visualProps.lightEmission = other.visualProps.lightEmission;
            visualProps.lightOpacity = other.visualProps.lightOpacity;
            interactionProps.canPlaceOn = other.interactionProps.canPlaceOn;
            interactionProps.canWalkThrough = other.interactionProps.canWalkThrough;
        }
        return *this;
    }
};
