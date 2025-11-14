package command

import (
	"github.com/mrhaoxx/go-mc/data/packetid"
	pk "github.com/mrhaoxx/go-mc/net/packet"
)

type Client interface {
	SendPacket(p pk.Packet)
}

// ClientJoin implement server.Component for Graph
func (g *Graph) ClientJoin(client Client) {
	client.SendPacket(pk.Marshal(
		packetid.ClientboundCommands, g,
	))
}
