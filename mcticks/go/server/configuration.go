package server

import (
	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/data/packetid"
	"github.com/mrhaoxx/go-mc/net"
	"github.com/mrhaoxx/go-mc/registry"

	pk "github.com/mrhaoxx/go-mc/net/packet"
)

type ConfigHandler interface {
	AcceptConfig(conn *net.Conn) error
}

type Configurations struct {
	Registries registry.Registries
}

var registers []string = []string{
	"minecraft:chat_type",
	"minecraft:dimension_type",
	"minecraft:damage_type",
	// "minecraft:trim_material",
	// "minecraft:trim_pattern",
	"minecraft:worldgen/biome",
	"minecraft:wolf_variant",
	"minecraft:painting_variant",
	// "minecraft:banner_pattern",
	// "minecraft:enchantment",
	// "minecraft:jukebox_song",
}

func (c *Configurations) AcceptConfig(conn *net.Conn) error {

	for _, registry := range registers {
		err := conn.WritePacket(pk.Marshal(
			packetid.ClientboundConfigRegistryData,
			pk.Identifier(registry),
			c.Registries.RegistryEncoder(registry),
		))
		if err != nil {
			return err
		}
	}

	err := conn.WritePacket(pk.Marshal(
		packetid.ClientboundConfigFinishConfiguration,
	))
	return err
}

type ConfigFailErr struct {
	reason chat.Message
}

func (c ConfigFailErr) Error() string {
	return "config error: " + c.reason.ClearString()
}
