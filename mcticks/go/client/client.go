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
	"sync"

	"go.uber.org/zap"

	"github.com/mrhaoxx/go-mc/data/packetid"
	"github.com/mrhaoxx/go-mc/data/registryid"
	"github.com/mrhaoxx/go-mc/level/block"
	"github.com/mrhaoxx/go-mc/net"
	pk "github.com/mrhaoxx/go-mc/net/packet"
	"github.com/mrhaoxx/go-mc/net/queue"
	"github.com/mrhaoxx/go-mc/server"
	"github.com/mrhaoxx/go-mc/world"
)

type Client struct {
	log      *zap.Logger
	conn     *net.Conn
	player   *world.Player
	world    *world.World
	queue    server.PacketQueue
	handlers []PacketHandler

	lastUpdatedChunks map[[2]int32]int32

	muLastUpdate sync.RWMutex
	// pointer to the Player.Input
	*world.Inputs
}

type PacketHandler func(p pk.Packet, c *Client) error

func New(log *zap.Logger, conn *net.Conn, player *world.Player, world *world.World) *Client {
	return &Client{
		log:               log,
		conn:              conn,
		player:            player,
		world:             world,
		queue:             queue.NewChannelQueue[pk.Packet](2560),
		handlers:          defaultHandlers[:],
		Inputs:            &player.Inputs,
		lastUpdatedChunks: make(map[[2]int32]int32),
	}
}

func (c *Client) Start() {
	stopped := make(chan struct{}, 2)
	done := func() {
		stopped <- struct{}{}
	}
	// Exit when any error is thrown
	go c.startSend(done)
	go c.startReceive(done)
	<-stopped
}

func (c *Client) startSend(done func()) {
	defer done()
	for {
		p, ok := c.queue.Pull()
		if !ok {
			return
		}
		err := c.conn.WritePacket(p)
		if err != nil {
			c.log.Debug("Send packet fail", zap.Error(err))
			return
		}
		if packetid.ClientboundPacketID(p.ID) == packetid.ClientboundDisconnect {
			return
		}
		// fmt.Println("Client: Sent packet", p.ID, len(p.Data))
	}
}

func (c *Client) startReceive(done func()) {
	defer done()
	var packet pk.Packet
	for {
		err := c.conn.ReadPacket(&packet)
		if err != nil {
			c.log.Debug("Receive packet fail", zap.Error(err))
			return
		}
		if packet.ID < 0 || packet.ID >= int32(len(c.handlers)) {
			c.log.Debug("Invalid packet id", zap.Int32("id", packet.ID), zap.Int("len", len(packet.Data)))
			return
		}
		if handler := c.handlers[packet.ID]; handler != nil {
			err = handler(packet, c)
			if err != nil {
				c.log.Error("Handle packet error", zap.Int32("id", packet.ID), zap.Error(err))
				return
			}
		} else {
			c.log.Info("Unhandled packet id", zap.Int32("id", packet.ID), zap.Int("len", len(packet.Data)), zap.String("type", string(packetid.ServerboundPacketID(packet.ID).String())))
		}
	}
}

func (c *Client) AddHandler(id packetid.ServerboundPacketID, handler PacketHandler) {
	c.handlers[id] = handler
}
func (c *Client) GetPlayer() *world.Player { return c.player }

// itemIDToBlockState converts item ID to block state ID
// For most blocks, the item ID matches the block state ID
func itemIDToBlockState(itemID int32) block.Block {
	var name = registryid.Item[itemID]
	var blockID = block.FromID[name]
	return blockID
}

func nop(p pk.Packet, c *Client) error { return nil }

var defaultHandlers = [packetid.ServerboundPacketIDGuard]PacketHandler{
	packetid.ServerboundAcceptTeleportation: clientAcceptTeleportation,
	// packetid.ServerboundClientInformation:    clientInformation,
	packetid.ServerboundMovePlayerPos:        clientMovePlayerPos,
	packetid.ServerboundMovePlayerPosRot:     clientMovePlayerPosRot,
	packetid.ServerboundMovePlayerRot:        clientMovePlayerRot,
	packetid.ServerboundMovePlayerStatusOnly: clientMovePlayerStatusOnly,
	packetid.ServerboundMoveVehicle:          clientMoveVehicle,
	packetid.ServerboundChatCommand:          clientCommand,
	packetid.ServerboundSwing:                clientSwing,
	packetid.ServerboundUseItemOn:            clientUseItemOn,
	packetid.ServerboundPlayerAction:         clientPlayerAction,
	packetid.ServerboundSetCarriedItem:       clientSetCarriedItem,
	packetid.ServerboundSetCreativeModeSlot:  clientSetCreativeModeSlot,
	packetid.ServerboundClientTickEnd:        nop, // nop
	packetid.ServerboundPlayerInput:          nop, // nop
	packetid.ServerboundCustomPayload:        nop, // nop
	packetid.ServerboundPlayerAbilities:      nop, // nop
	packetid.ServerboundPlayerCommand:        nop, // nop

}

func clientInformation(p pk.Packet, client *Client) error {
	var info world.ClientInfo
	if err := p.Scan(&info); err != nil {
		return err
	}
	client.Inputs.Lock()
	client.Inputs.ClientInfo = info
	client.Inputs.Unlock()
	return nil
}
