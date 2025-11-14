#include "Block.h"

BlockRegistry globalBlockRegistry;

// 注册需要使用的方块，没定义的方块会使用默认方块
struct BlockTypeInitializer {
    BlockTypeInitializer() {
        // 空气
        globalBlockRegistry.registerBlock(0, BlockInfo(
            "Air",
            BlockInfo::VisualProperties(0, 0),
            BlockInfo::InteractionProperties(false, true)
        ));
        // 石头
        globalBlockRegistry.registerBlock(1, BlockInfo(
            "Stone",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 泥土
        globalBlockRegistry.registerBlock(10, BlockInfo(
            "Dirt",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 黑曜石
        globalBlockRegistry.registerBlock(2397, BlockInfo(
            "Obsidian",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 圆石
        globalBlockRegistry.registerBlock(14, BlockInfo(
            "Cobblestone",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 基岩
        globalBlockRegistry.registerBlock(85, BlockInfo(
            "Bedrock",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 草方块
        globalBlockRegistry.registerBlock(8, BlockInfo(
            "Grass Block",
            BlockInfo::VisualProperties(0, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 水
        globalBlockRegistry.registerBlock(86, BlockInfo(
            "Water",
            BlockInfo::VisualProperties(0, 2),
            BlockInfo::InteractionProperties(false, true)
        ), 16); // 16 stages for water

        // 熔岩
        globalBlockRegistry.registerBlock(102, BlockInfo(
            "Lava",
            BlockInfo::VisualProperties(15, 15),
            BlockInfo::InteractionProperties(false, true)
        ), 16);
        // 火
        globalBlockRegistry.registerBlock(2403, BlockInfo(
            "Fire",
            BlockInfo::VisualProperties(15, 15),
            BlockInfo::InteractionProperties(true, true)
        ));
        // 萤石
        globalBlockRegistry.registerBlock(6032, BlockInfo(
            "Glowstone",
            BlockInfo::VisualProperties(15, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 海晶石
        globalBlockRegistry.registerBlock(11603, BlockInfo(
            "SeaLantern",
            BlockInfo::VisualProperties(15, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 末地烛
        globalBlockRegistry.registerBlock(13351, BlockInfo(
            "EndRod",
            BlockInfo::VisualProperties(14, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 灯笼
        globalBlockRegistry.registerBlock(19517, BlockInfo(
            "Lantern",
            BlockInfo::VisualProperties(15, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 火把
        globalBlockRegistry.registerBlock(2398, BlockInfo(
            "Torch",
            BlockInfo::VisualProperties(14, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 红石火把
        globalBlockRegistry.registerBlock(5907, BlockInfo(
            "RedstoneTorch",
            BlockInfo::VisualProperties(7, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
        // 灵魂火把
        globalBlockRegistry.registerBlock(6027, BlockInfo(
            "SoulTorch",
            BlockInfo::VisualProperties(10, 15),
            BlockInfo::InteractionProperties(true, false)
        ));
    }
} blockTypeInitializer; 