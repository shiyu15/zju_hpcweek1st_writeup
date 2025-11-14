package client

import (
	"fmt"

	"github.com/mrhaoxx/go-mc/hpcworld"
	pk "github.com/mrhaoxx/go-mc/net/packet"
)

func clientPlayerAction(p pk.Packet, c *Client) error {
	var (
		status pk.VarInt
		pos    pk.Position
		face   pk.VarInt
		seq    pk.VarInt
	)
	if err := p.Scan(&status, &pos, &face, &seq); err != nil {
		// Fallback: without sequence
		if err2 := p.Scan(&status, &pos, &face); err2 != nil {
			return err
		}
	}
	fmt.Println("Client: Player action", status, pos, face, seq)
	// Only handle finish-destroy to actually remove the block.
	if int(status) == 2 || int(status) == 0 { // Stop/Finish digging
		x, y, z := pos.X, pos.Y, pos.Z
		hpcworld.SetBlock(int32(x), int32(y), int32(z), 0)
	}
	return nil
}
