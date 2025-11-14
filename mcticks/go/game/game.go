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

package game

import (
	"context"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/client"
	"github.com/mrhaoxx/go-mc/data/packetid"
	"github.com/mrhaoxx/go-mc/judge"
	"github.com/mrhaoxx/go-mc/net"
	pk "github.com/mrhaoxx/go-mc/net/packet"
	"github.com/mrhaoxx/go-mc/server"
	"github.com/mrhaoxx/go-mc/world"
	"github.com/mrhaoxx/go-mc/yggdrasil/user"
)

type Game struct {
	log *zap.Logger

	config     Config
	serverInfo *server.PingInfo

	overworld *world.World

	globalChat globalChat
	*playerList
}

// World returns the primary world instance managed by the game.
func (g *Game) World() *world.World {
	return g.overworld
}

func NewGame(log *zap.Logger, config Config, pingList *server.PlayerList, serverInfo *server.PingInfo) *Game {
	overworld := world.New(
		log.Named("overworld"),
		world.Config{
			ViewDistance: config.ViewDistance,
			// SpawnAngle:    lv.Data.SpawnAngle,
			SpawnPosition: [3]int32{0, 100, 0},
		},
	)
	overworld.SetJudgeProvider(judge.NewProvider())

	keepAlive := server.NewKeepAlive()
	pl := playerList{pingList: pingList, keepAlive: keepAlive}
	keepAlive.AddPlayerDelayUpdateHandler(func(c server.KeepAliveClient, latency time.Duration) {
		pl.updateLatency(c.(*client.Client), latency)
	})
	go keepAlive.Run(context.TODO())

	g := globalChat{
		log:           log.Named("chat"),
		players:       &pl,
		chatTypeCodec: &world.NetworkCodec.ChatType,
	}

	return &Game{
		log: log.Named("game"),

		config:     config,
		serverInfo: serverInfo,

		overworld: overworld,

		globalChat: g,
		playerList: &pl,
	}
}

// AcceptPlayer will be called in an independent goroutine when new player login
func (g *Game) AcceptPlayer(name string, id uuid.UUID, profilePubKey *user.PublicKey, properties []user.Property, protocol int32, conn *net.Conn) {
	logger := g.log.With(
		zap.String("name", name),
		zap.String("uuid", id.String()),
		zap.Int32("protocol", protocol),
	)

	p := &world.Player{
		Entity: world.Entity{
			EntityID: world.NewEntityID(),
			Position: [3]float64{0, 100, 0},
			Rotation: [2]float32{},
		},
		Name:           name,
		UUID:           id,
		PubKey:         profilePubKey,
		Properties:     properties,
		Gamemode:       1,
		ChunkPos:       [3]int32{48 >> 4, 64 >> 4, 35 >> 4},
		EntitiesInView: make(map[int32]*world.Entity),
		ViewDistance:   8,
	}

	c := client.New(logger, conn, p, g.overworld)

	logger.Info("Player join", zap.Int32("eid", p.EntityID))
	defer logger.Info("Player left")

	c.SendLogin(g.overworld, p)
	c.SendGameEvent(pk.UnsignedByte(13), pk.Float(0))

	g.globalChat.broadcastSystemChat(chat.Text("player joined "+p.Name), false)
	defer g.globalChat.broadcastSystemChat(chat.Text("Player left "+p.Name), false)

	c.AddHandler(packetid.ServerboundChat, g.globalChat.Handle)

	g.playerList.addPlayer(c, p)
	defer g.playerList.removePlayer(c)

	c.SendPlayerPosition(p.Position, p.Rotation)
	g.overworld.AddPlayer(c, p, g.config.PlayerChunkLoadingLimiter.Limiter())
	defer g.overworld.RemovePlayer(c, p)

	c.SendPacket(packetid.ClientboundUpdateTags, pk.Array(defaultTags))
	c.SendSetDefaultSpawnPosition(g.overworld.SpawnPositionAndAngle())
	c.Start()
}
