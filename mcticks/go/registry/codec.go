package registry

import (
	"io"
	"reflect"

	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/nbt"
	pk "github.com/mrhaoxx/go-mc/net/packet"
)

type Registries struct {
	ChatType        Registry[ChatType]       `registry:"minecraft:chat_type"`
	DamageType      Registry[DamageType]     `registry:"minecraft:damage_type"`
	DimensionType   Registry[Dimension]      `registry:"minecraft:dimension_type"`
	TrimMaterial    Registry[nbt.RawMessage] `registry:"minecraft:trim_material"`
	TrimPattern     Registry[nbt.RawMessage] `registry:"minecraft:trim_pattern"`
	WorldGenBiome   Registry[any]            `registry:"minecraft:worldgen/biome"`
	Wolfvariant     Registry[any]            `registry:"minecraft:wolf_variant"`
	PaintingVariant Registry[any]            `registry:"minecraft:painting_variant"`
	BannerPattern   Registry[nbt.RawMessage] `registry:"minecraft:banner_pattern"`
	Enchantment     Registry[nbt.RawMessage] `registry:"minecraft:enchantment"`
	JukeboxSong     Registry[nbt.RawMessage] `registry:"minecraft:jukebox_song"`
}

type MonsterSpawnLightLevel struct {
	Max_inclusive int32  `nbt:"max_inclusive"`
	Min_inclusive int32  `nbt:"min_inclusive"`
	Type_         string `nbt:"type"`
}

func NewNetworkCodec() Registries {

	def := NewRegistry[Dimension]()

	def.Put("minecraft:overworld", Dimension{
		FixedTime:          0,
		HasSkylight:        true,
		HasCeiling:         false,
		Ultrawarm:          false,
		Natural:            true,
		CoordinateScale:    1.0,
		BedWorks:           true,
		RespawnAnchorWorks: 0,
		MinY:               -64,
		Height:             384,
		LogicalHeight:      384,
		InfiniteBurn:       "#minecraft:infiniburn_overworld",
		Effects:            "minecraft:overworld",
		AmbientLight:       0.0,
		PiglinSafe:         0.0,
		HasRaids:           1,
		MonsterSpawnLightLevel: MonsterSpawnLightLevel{
			Min_inclusive: 0,
			Max_inclusive: 7,
			Type_:         "minecraft:uniform",
		},
		MonsterSpawnBlockLightLimit: 0,
	})

	return Registries{
		ChatType:        NewRegistry[ChatType](),
		DamageType:      NewRegistry[DamageType](),
		DimensionType:   def,
		TrimMaterial:    NewRegistry[nbt.RawMessage](),
		TrimPattern:     NewRegistry[nbt.RawMessage](),
		WorldGenBiome:   NewRegistry[any](),
		Wolfvariant:     NewRegistry[any](),
		PaintingVariant: NewRegistry[any](),
		BannerPattern:   NewRegistry[nbt.RawMessage](),
		Enchantment:     NewRegistry[nbt.RawMessage](),
		JukeboxSong:     NewRegistry[nbt.RawMessage](),
	}
}

type ChatType struct {
	Chat      chat.Decoration `nbt:"chat" json:"chat"`
	Narration chat.Decoration `nbt:"narration" json:"narration"`
}

type DamageType struct {
	MessageID        string  `nbt:"message_id" json:"message_id"`
	Scaling          string  `nbt:"scaling" json:"scaling"`
	Exhaustion       float32 `nbt:"exhaustion" json:"exhaustion"`
	Effects          string  `nbt:"effects,omitempty"`
	DeathMessageType string  `nbt:"death_message_type,omitempty"`
}

type Dimension struct {
	MonsterSpawnLightLevel MonsterSpawnLightLevel `nbt:"monster_spawn_light_level"` // Tag_Int or {type:"minecraft:uniform", value:{min_inclusive: Tag_Int, max_inclusive: Tag_Int}}

	FixedTime          int64   `nbt:"fixed_time,omitempty"`
	HasSkylight        bool    `nbt:"has_skylight"`
	HasCeiling         bool    `nbt:"has_ceiling"`
	Ultrawarm          bool    `nbt:"ultrawarm"`
	Natural            bool    `nbt:"natural"`
	CoordinateScale    float64 `nbt:"coordinate_scale"`
	BedWorks           bool    `nbt:"bed_works"`
	RespawnAnchorWorks byte    `nbt:"respawn_anchor_works"`
	MinY               int32   `nbt:"min_y"`
	Height             int32   `nbt:"height"`
	LogicalHeight      int32   `nbt:"logical_height"`
	InfiniteBurn       string  `nbt:"infiniburn"`
	Effects            string  `nbt:"effects"`
	AmbientLight       float64 `nbt:"ambient_light"`

	PiglinSafe                  byte  `nbt:"piglin_safe"`
	HasRaids                    byte  `nbt:"has_raids"`
	MonsterSpawnBlockLightLimit int32 `nbt:"monster_spawn_block_light_limit"`
}

type RegistryCodec interface {
	pk.FieldDecoder
	ReadTagsFrom(r io.Reader) (int64, error)
}

func (c *Registries) Registry(id string) RegistryCodec {
	codecVal := reflect.ValueOf(c).Elem()
	codecTyp := codecVal.Type()
	numField := codecVal.NumField()
	for i := 0; i < numField; i++ {
		registryID, ok := codecTyp.Field(i).Tag.Lookup("registry")
		if !ok {
			continue
		}
		if registryID == id {
			return codecVal.Field(i).Addr().Interface().(RegistryCodec)
		}
	}
	return nil
}

func (c *Registries) RegistryEncoder(id string) pk.FieldEncoder {
	codecVal := reflect.ValueOf(c).Elem()
	codecTyp := codecVal.Type()
	numField := codecVal.NumField()
	for i := 0; i < numField; i++ {
		registryID, ok := codecTyp.Field(i).Tag.Lookup("registry")
		if !ok {
			continue
		}
		if registryID == id {
			return codecVal.Field(i).Addr().Interface().(pk.FieldEncoder)
		}
	}
	return nil
}
