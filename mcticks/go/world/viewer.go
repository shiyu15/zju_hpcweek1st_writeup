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
	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/level"
)

type Client interface {
	ChunkViewer
	EntityViewer
	SendDisconnect(reason chat.Message)
	SendPlayerPosition(pos [3]float64, rot [2]float32) (teleportID int32)
	SendSetChunkCacheCenter(chunkPos [2]int32)
	SendSetPlayerInventorySlot(slot int32, stack *ItemStack)
	SendSystemChat(msg chat.Message, overlay bool)
	SendSetTime(worldAge int64, timeOfDay int64, doDaylightCycle bool)
}

type ChunkViewer interface {
	ViewChunkLoad(pos level.ChunkPos, c *LoadedChunk)
	ViewChunkUnload(pos level.ChunkPos)
}

type EntityViewer interface {
	ViewAddPlayer(p *Player)
	ViewAddEntity(e *Entity, typeName string)
	ViewRemoveEntities(entityIDs []int32)
	ViewMoveEntityPos(id int32, delta [3]int16, onGround bool)
	ViewMoveEntityPosAndRot(id int32, delta [3]int16, rot [2]int8, onGround bool)
	ViewMoveEntityRot(id int32, rot [2]int8, onGround bool)
	ViewRotateHead(id int32, yaw int8)
	ViewTeleportEntity(id int32, pos [3]float64, rot [2]int8, onGround bool)
	ViewSetEntityMotion(id int32, velocity [3]float64)
	ViewAnimate(id int32, animation byte)
}
