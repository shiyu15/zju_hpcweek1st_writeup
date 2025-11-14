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

package client

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"unsafe"

	"go.uber.org/zap"

	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/data/packetid"
	"github.com/mrhaoxx/go-mc/data/registryid"
	"github.com/mrhaoxx/go-mc/level"
	"github.com/mrhaoxx/go-mc/nbt"
	pk "github.com/mrhaoxx/go-mc/net/packet"
	"github.com/mrhaoxx/go-mc/world"
)

func (c *Client) SendPacket(id packetid.ClientboundPacketID, fields ...pk.FieldEncoder) {
	var buffer bytes.Buffer

	// Write the packet fields
	for i := range fields {
		if _, err := fields[i].WriteTo(&buffer); err != nil {
			c.log.Panic("Marshal packet error", zap.Error(err))
		}
	}

	// Send the packet data
	c.queue.Push(pk.Packet{
		ID:   int32(id),
		Data: buffer.Bytes(),
	})
}

func (c *Client) SendKeepAlive(id int64) {
	c.SendPacket(packetid.ClientboundKeepAlive, pk.Long(id))
}

func (c *Client) SendDisconnect(reason chat.Message) {
	c.log.Debug("Disconnect player", zap.String("reason", reason.ClearString()))
	c.SendPacket(packetid.ClientboundDisconnect, reason)
}

func (c *Client) SendLogin(w *world.World, p *world.Player) {
	zap.L().Info("SendLogin", zap.Int32("eid", p.EntityID), zap.Int32("viewDistance", p.ViewDistance))
	hashedSeed := w.HashedSeed()
	c.SendPacket(
		packetid.ClientboundLogin,
		pk.Int(p.EntityID),
		pk.Boolean(false), // Is Hardcore
		pk.Array([]pk.Identifier{
			pk.Identifier(w.Name()),
		}), // Dimension Name
		pk.VarInt(20),                        // Max players (ignored by client)
		pk.VarInt(p.ViewDistance),            // View Distance
		pk.VarInt(p.ViewDistance),            // Simulation Distance
		pk.Boolean(false),                    // Reduced Debug Info
		pk.Boolean(true),                     // Enable respawn screen
		pk.Boolean(false),                    // Do Limit Crafting
		pk.VarInt(0),                         // World Info Dimension Type
		pk.Identifier("minecraft:overworld"), // World Info Dimension Name
		pk.Long(binary.BigEndian.Uint64(hashedSeed[:8])), // World Info Hashed Seed
		pk.Byte(1),        // World Info Gamemode
		pk.Byte(0),        // World Info Previous Gamemode
		pk.Boolean(false), // World Info Is Debug
		pk.Boolean(false), // World Info Is Flat
		pk.Boolean(false), // World Info Has Last Death Location
		pk.VarInt(40),
		pk.VarInt(40),
		pk.Boolean(true),
	)
}

func (c *Client) SendGameEvent(event pk.UnsignedByte, value pk.Float) {
	c.SendPacket(
		packetid.ClientboundGameEvent,
		pk.UnsignedByte(event),
		pk.Float(value),
	)
}

// func (c *Client) SendServerData(motd *chat.Message, favIcon string, enforceSecureProfile bool) {
// 	c.SendPacket(
// 		packetid.ClientboundServerData,
// 		motd,
// 		pk.Option[pk.String, *pk.String]{
// 			Has: false,
// 			Val: pk.String(favIcon),
// 		},
// 		// pk.Boolean(enforceSecureProfile),
// 	)
// }

// Actions of [SendPlayerInfoUpdate]
const (
	PlayerInfoAddPlayer = iota
	PlayerInfoInitializeChat
	PlayerInfoUpdateGameMode
	PlayerInfoUpdateListed
	PlayerInfoUpdateLatency
	PlayerInfoUpdateDisplayName
	// PlayerInfoEnumGuard is the number of the enums
	PlayerInfoEnumGuard
)

func NewPlayerInfoAction(actions ...int) pk.FixedBitSet {
	enumSet := pk.NewFixedBitSet(PlayerInfoEnumGuard)
	for _, action := range actions {
		enumSet.Set(action, true)
	}
	return enumSet
}

func (c *Client) SendPlayerInfoUpdate(actions pk.FixedBitSet, players []*world.Player) {
	var buf bytes.Buffer
	_, _ = actions.WriteTo(&buf)
	_, _ = pk.VarInt(len(players)).WriteTo(&buf)
	for _, player := range players {
		_, _ = pk.UUID(player.UUID).WriteTo(&buf)
		if actions.Get(PlayerInfoAddPlayer) {
			_, _ = pk.String(player.Name).WriteTo(&buf)
			_, _ = pk.Array(player.Properties).WriteTo(&buf)
		}
		if actions.Get(PlayerInfoInitializeChat) {
			panic("not yet support InitializeChat")
		}
		if actions.Get(PlayerInfoUpdateGameMode) {
			_, _ = pk.VarInt(player.Gamemode).WriteTo(&buf)
		}
		if actions.Get(PlayerInfoUpdateListed) {
			_, _ = pk.Boolean(true).WriteTo(&buf)
		}
		if actions.Get(PlayerInfoUpdateLatency) {
			_, _ = pk.VarInt(player.Latency.Milliseconds()).WriteTo(&buf)
		}
		if actions.Get(PlayerInfoUpdateDisplayName) {
			panic("not yet support DisplayName")
		}
	}
	c.queue.Push(pk.Packet{
		ID:   int32(packetid.ClientboundPlayerInfoUpdate),
		Data: buf.Bytes(),
	})
}

func (c *Client) SendPlayerInfoRemove(players []*world.Player) {
	var buff bytes.Buffer

	if _, err := pk.VarInt(len(players)).WriteTo(&buff); err != nil {
		c.log.Panic("Marshal packet error", zap.Error(err))
	}
	for _, p := range players {
		if _, err := pk.UUID(p.UUID).WriteTo(&buff); err != nil {
			c.log.Panic("Marshal packet error", zap.Error(err))
		}
	}

	c.queue.Push(pk.Packet{
		ID:   int32(packetid.ClientboundPlayerInfoRemove),
		Data: buff.Bytes(),
	})
}

func (c *Client) SendLevelChunkWithLight(pos level.ChunkPos, chunk *world.LoadedChunk) {

	c.muLastUpdate.RLock()
	ll, ok := c.lastUpdatedChunks[pos]
	if ok && ll == chunk.Lastrefresh {
		c.muLastUpdate.RUnlock()
		return
	}
	c.muLastUpdate.RUnlock()
	c.muLastUpdate.Lock()
	c.lastUpdatedChunks[pos] = chunk.Lastrefresh
	c.muLastUpdate.Unlock()

	c.SendPacket(packetid.ClientboundLevelChunkWithLight, pos, chunk.Cvt)
}

func (c *Client) SendForgetLevelChunk(pos level.ChunkPos) {
	delete(c.lastUpdatedChunks, pos)
	c.SendPacket(packetid.ClientboundForgetLevelChunk, pos)
}

func (c *Client) SendAddPlayer(p *world.Player) {
	// Spawn the player entity for viewers.
	// Use AddEntity with entity type set to "minecraft:player".
	yaw := int8(p.Rotation[0] * 256 / 360)
	pitch := int8(p.Rotation[1] * 256 / 360)
	// lookup entity type id
	var typeID int32 = -1
	for i, name := range registryid.EntityType {
		if name == "minecraft:player" {
			typeID = int32(i)
			break
		}
	}
	if typeID < 0 {
		zap.L().Warn("entity type not found", zap.String("type", "minecraft:player"))
		return
	}
	c.SendPacket(
		packetid.ClientboundAddEntity,
		pk.VarInt(p.EntityID),
		pk.UUID(p.UUID),
		pk.VarInt(typeID),
		pk.Double(p.Position[0]),
		pk.Double(p.Position[1]),
		pk.Double(p.Position[2]),
		pk.Angle(yaw),
		pk.Angle(pitch),
		pk.Angle(yaw), // head yaw
		pk.VarInt(0),
		pk.Short(0), // vel x
		pk.Short(0), // vel y
		pk.Short(0), // vel z
	)
}

func (c *Client) SendMoveEntitiesPos(eid int32, delta [3]int16, onGround bool) {
	zap.L().Info("SendMoveEntitiesPos", zap.Int32("eid", eid), zap.Int16("delta[0]", delta[0]), zap.Int16("delta[1]", delta[1]), zap.Int16("delta[2]", delta[2]), zap.Bool("onGround", onGround))
	c.SendPacket(
		packetid.ClientboundMoveEntityPos,
		pk.VarInt(eid),
		pk.Short(delta[0]),
		pk.Short(delta[1]),
		pk.Short(delta[2]),
		pk.Boolean(onGround),
	)
}

func (c *Client) SendMoveEntitiesPosAndRot(eid int32, delta [3]int16, rot [2]int8, onGround bool) {
	zap.L().Info("SendMoveEntitiesPosAndRot", zap.Int32("eid", eid), zap.Int16("delta[0]", delta[0]), zap.Int16("delta[1]", delta[1]), zap.Int16("delta[2]", delta[2]), zap.Int8("rot[0]", rot[0]), zap.Int8("rot[1]", rot[1]), zap.Bool("onGround", onGround))
	c.SendPacket(
		packetid.ClientboundMoveEntityPosRot,
		pk.VarInt(eid),
		pk.Short(delta[0]),
		pk.Short(delta[1]),
		pk.Short(delta[2]),
		pk.Angle(rot[0]),
		pk.Angle(rot[1]),
		pk.Boolean(onGround),
	)
}

func (c *Client) SendMoveEntitiesRot(eid int32, rot [2]int8, onGround bool) {
	zap.L().Info("SendMoveEntitiesRot", zap.Int32("eid", eid), zap.Int8("rot[0]", rot[0]), zap.Int8("rot[1]", rot[1]), zap.Bool("onGround", onGround))
	c.SendPacket(
		packetid.ClientboundMoveEntityRot,
		pk.VarInt(eid),
		pk.Angle(rot[0]),
		pk.Angle(rot[1]),
		pk.Boolean(onGround),
	)
}

func (c *Client) SendRotateHead(eid int32, yaw int8) {
	c.SendPacket(
		packetid.ClientboundRotateHead,
		pk.VarInt(eid),
		pk.Angle(yaw),
	)
}

func (c *Client) SendTeleportEntity(eid int32, pos [3]float64, rot [2]int8, onGround bool) {
	zap.L().Info("SendTeleportEntity", zap.Int32("eid", eid), zap.Float64("pos[0]", pos[0]), zap.Float64("pos[1]", pos[1]), zap.Float64("pos[2]", pos[2]), zap.Int8("rot[0]", rot[0]), zap.Int8("rot[1]", rot[1]), zap.Bool("onGround", onGround))
	c.SendPacket(
		packetid.ClientboundTeleportEntity,
		pk.VarInt(eid),
		pk.Double(pos[0]),
		pk.Double(pos[1]),
		pk.Double(pos[2]),
		pk.Angle(rot[0]),
		pk.Angle(rot[1]),
		pk.Boolean(onGround),
	)
}

var teleportCounter atomic.Int32

func (c *Client) SendPlayerPosition(pos [3]float64, rot [2]float32) (teleportID int32) {
	teleportID = teleportCounter.Add(1)

	fmt.Println("SendPlayerPosition", teleportID, pos[0], pos[1], pos[2], rot[0], rot[1])
	c.SendPacket(
		packetid.ClientboundPlayerPosition,
		pk.VarInt(teleportID),
		pk.Double(pos[0]),
		pk.Double(pos[1]),
		pk.Double(pos[2]),
		pk.Double(0),
		pk.Double(0),
		pk.Double(0),
		pk.Float(rot[0]),
		pk.Float(rot[1]),
		pk.Int(0), // Absolute
	)
	return
}

func (c *Client) SendSetDefaultSpawnPosition(xyz [3]int32, angle float32) {
	c.SendPacket(
		packetid.ClientboundSetDefaultSpawnPosition,
		pk.Position{X: int(xyz[0]), Y: int(xyz[1]), Z: int(xyz[2])},
		pk.Float(angle),
	)
	return
}

func (c *Client) SendSetPlayerInventorySlot(slot int32, stack *world.ItemStack) {
	localStack := &ItemStack{
		ItemID: stack.ItemID,
		Count:  stack.Count,
	}
	fields := []pk.FieldEncoder{pk.VarInt(slot)}
	fields = append(fields, localStack.encodeFields()...)
	c.SendPacket(packetid.ClientboundSetPlayerInventory, fields...)
}

func (c *Client) SendRemoveEntities(entityIDs []int32) {
	c.SendPacket(
		packetid.ClientboundRemoveEntities,
		pk.Array(*(*[]pk.VarInt)(unsafe.Pointer(&entityIDs))),
	)
}

func (c *Client) SendSystemChat(msg chat.Message, overlay bool) {

	snbt := nbt.StringifiedMessage(fmt.Sprintf(`{text:%q}`, msg.String()))
	c.SendPacket(packetid.ClientboundSystemChat,
		pk.NBT(snbt),
		pk.Boolean(overlay),
	)
}

func (c *Client) SendSetChunkCacheCenter(chunkPos [2]int32) {
	c.SendPacket(
		packetid.ClientboundSetChunkCacheCenter,
		pk.VarInt(chunkPos[0]),
		pk.VarInt(chunkPos[1]),
	)
}
func (c *Client) SendSetTime(worldAge int64, timeOfDay int64, doDaylightCycle bool) {
	c.SendPacket(
		packetid.ClientboundSetTime,
		pk.Long(worldAge),
		pk.Long(timeOfDay),
		pk.Boolean(doDaylightCycle),
	)
}

func (c *Client) ViewChunkLoad(pos level.ChunkPos, chunk *world.LoadedChunk) {
	c.SendLevelChunkWithLight(pos, chunk)
}
func (c *Client) ViewChunkUnload(pos level.ChunkPos)   { c.SendForgetLevelChunk(pos) }
func (c *Client) ViewAddPlayer(p *world.Player)        { c.SendAddPlayer(p) }
func (c *Client) ViewRemoveEntities(entityIDs []int32) { c.SendRemoveEntities(entityIDs) }
func (c *Client) ViewMoveEntityPos(id int32, delta [3]int16, onGround bool) {
	c.SendMoveEntitiesPos(id, delta, onGround)
}

func (c *Client) ViewMoveEntityPosAndRot(id int32, delta [3]int16, rot [2]int8, onGround bool) {
	c.SendMoveEntitiesPosAndRot(id, delta, rot, onGround)
}

func (c *Client) ViewMoveEntityRot(id int32, rot [2]int8, onGround bool) {
	c.SendMoveEntitiesRot(id, rot, onGround)
}

func (c *Client) ViewRotateHead(id int32, yaw int8) {
	c.SendRotateHead(id, yaw)
}

func (c *Client) ViewTeleportEntity(id int32, pos [3]float64, rot [2]int8, onGround bool) {
	c.SendTeleportEntity(id, pos, rot, onGround)
}

func (c *Client) ViewSetEntityMotion(id int32, velocity [3]float64) {
	c.SendSetEntityMotion(id, velocity)
}

// SendSetEntityMotion sets an entity's velocity (blocks/tick scaled by 8000).
func (c *Client) SendSetEntityMotion(eid int32, velocity [3]float64) {
	clamp := func(x int32) int16 {
		if x > 32767 {
			return 32767
		}
		if x < -32768 {
			return -32768
		}
		return int16(x)
	}
	vx := clamp(int32(velocity[0] * 8000))
	vy := clamp(int32(velocity[1] * 8000))
	vz := clamp(int32(velocity[2] * 8000))
	c.SendPacket(
		packetid.ClientboundSetEntityMotion,
		pk.VarInt(eid),
		pk.Short(vx),
		pk.Short(vy),
		pk.Short(vz),
	)
}

// ViewAnimate sends an animation for an entity to this client.
func (c *Client) ViewAnimate(id int32, animation byte) {
	c.SendAnimate(id, animation)
}

// SendAnimate emits ClientboundAnimate for the given entity and animation.
func (c *Client) SendAnimate(eid int32, animation byte) {
	c.SendPacket(
		packetid.ClientboundAnimate,
		pk.VarInt(eid),
		pk.UnsignedByte(animation),
	)
}

// ViewAddEntity spawns a generic entity for the viewer by registry name.
func (c *Client) ViewAddEntity(e *world.Entity, typeName string) {
	// Resolve entity type ID
	var typeID int32 = -1
	for i, name := range registryid.EntityType {
		if name == typeName {
			typeID = int32(i)
			break
		}
	}
	if typeID < 0 {
		zap.L().Warn("entity type not found", zap.String("type", typeName))
		return
	}
	yaw := int8(e.Rotation[0] * 256 / 360)
	pitch := int8(e.Rotation[1] * 256 / 360)
	c.SendPacket(
		packetid.ClientboundAddEntity,
		pk.VarInt(e.EntityID),
		pk.UUID(e.UUID),
		pk.VarInt(typeID),
		pk.Double(e.Position[0]),
		pk.Double(e.Position[1]),
		pk.Double(e.Position[2]),
		pk.Angle(pitch),
		pk.Angle(yaw),
		pk.Angle(yaw),
		pk.VarInt(0),
		pk.Short(0),
		pk.Short(0),
		pk.Short(0),
	)
}
