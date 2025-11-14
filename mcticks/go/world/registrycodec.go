// This file is part of go-mc/server project.
// Copyright (C) 2023.  Tnze
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

package world

import (
	_ "embed"
	"encoding/json"

	"github.com/mrhaoxx/go-mc/registry"
)

type JSONPaintingVariant struct {
	AssetIDF string `nbt:"asset_id", json:"asset_id"`
	Height   int32  `nbt:"height", json:"height"`
	Width    int32  `nbt:"width", json:"width"`
}

type JSONWolfVariant struct {
	Angry_texture string `nbt:"angry_texture", json:"angry_texture"`
	Biomes        string `nbt:"biomes", json:"biomes"`
	Wild_texture  string `nbt:"wild_texture", json:"wild_texture"`
	Tame_texture  string `nbt:"tame_texture", json:"tame_texture"`
}

type JSONMoodSound struct {
	Block_search_extent int32   `nbt:"block_search_extent", json:"block_search_extent"`
	Offset              float64 `nbt:"offset", json:"offset"`
	Sound               string  `nbt:"sound", json:"sound"`
	Tick_delay          int32   `nbt:"tick_delay", json:"tick_delay"`
}

type JSONMusic struct {
	Max_delay             int32  `nbt:"max_delay", json:"max_delay"`
	Min_delay             int32  `nbt:"min_delay", json:"min_delay"`
	Replace_current_music bool   `nbt:"replace_current_music", json:"replace_current_music"`
	Sound                 string `nbt:"sound", json:"sound"`
}

type JSONRegistry struct {
	PaintingVariant map[string]JSONPaintingVariant `json:"minecraft:painting_variant"`
	WolfVariant     map[string]JSONWolfVariant     `json:"minecraft:wolf_variant"`
	WorldGenBiome   map[string]JSONWorldGenBiome   `json:"minecraft:worldgen/biome"`
	DamageType      map[string]registry.DamageType `json:"minecraft:damage_type"`
	ChatType        map[string]registry.ChatType   `json:"minecraft:chat_type"`
}

type JSONEffects struct {
	Fog_color       int32 `nbt:"fog_color", json:"fog_color"`
	Sky_color       int32 `nbt:"sky_color", json:"sky_color"`
	Water_color     int32 `nbt:"water_color", json:"water_color"`
	Water_fog_color int32 `nbt:"water_fog_color", json:"water_fog_color"`
	// Foliage_color   int32 `nbt:"foliage_color", json:"foliage_color"`
	// Grass_color     int32 `nbt:"grass_color", json:"grass_color"`
	// Grass_color_modifier string        `nbt:"grass_color_modifier", json:"grass_color_modifier"`
	// Mood_sound JSONMoodSound `nbt:"mood_sound", json:"mood_sound"`
	// Music      JSONMusic     `nbt:"music", json:"music"`
}

type JSONWorldGenBiome struct {
	Downfall          float32     `nbt:"downfall", json:"downfall"`
	Temperature       float32     `nbt:"temperature", json:"temperature"`
	Effects           JSONEffects `nbt:"effects", json:"effects"`
	Has_precipitation bool        `nbt:"has_precipitation", json:"has_precipitation"`
	// Temperature_modifier string      `nbt:"temperature_modifier", json:"temperature_modifier"`
}

//go:embed 1\.21-registry/painting_variant.json
var paintingVariant []byte

//go:embed 1\.21-registry/wolf_variant.json
var wolfVariant []byte

//go:embed 1\.21-registry/biome.json
var biome []byte

//go:embed 1\.21-registry/damage_type.json
var damageType []byte

//go:embed 1\.21-registry/chat_type.json
var chatType []byte

var NetworkCodec registry.Registries = registry.NewNetworkCodec()

func init() {

	var jsonRegistry JSONRegistry
	if err := json.Unmarshal(paintingVariant, &jsonRegistry); err != nil {
		panic(err)
	}
	for key, value := range jsonRegistry.PaintingVariant {
		NetworkCodec.PaintingVariant.Put(key, value)
	}

	if err := json.Unmarshal(wolfVariant, &jsonRegistry); err != nil {
		panic(err)
	}
	for key, value := range jsonRegistry.WolfVariant {
		NetworkCodec.Wolfvariant.Put(key, value)
	}

	if err := json.Unmarshal(biome, &jsonRegistry); err != nil {
		panic(err)
	}
	for key, value := range jsonRegistry.WorldGenBiome {
		NetworkCodec.WorldGenBiome.Put(key, value)
	}

	if err := json.Unmarshal(damageType, &jsonRegistry); err != nil {
		panic(err)
	}
	for key, value := range jsonRegistry.DamageType {
		NetworkCodec.DamageType.Put(key, value)
	}

	if err := json.Unmarshal(chatType, &jsonRegistry); err != nil {
		panic(err)
	}
	for key, value := range jsonRegistry.ChatType {
		NetworkCodec.ChatType.Put(key, value)
	}
}
