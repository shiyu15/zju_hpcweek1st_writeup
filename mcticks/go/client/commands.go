package client

import (
	"fmt"
	"strings"

	pk "github.com/mrhaoxx/go-mc/net/packet"
)

func clientCommand(p pk.Packet, c *Client) error {
	var command pk.String
	if err := p.Scan(&command); err != nil {
		return err
	}
	raw := strings.TrimSpace(string(command))
	fmt.Println("command", raw)

	if raw == "" {
		c.world.EnqueueCommand(c, c.player, "", nil)
		return nil
	}

	fields := strings.Fields(raw)
	cmd := fields[0]
	args := append([]string(nil), fields[1:]...)

	c.world.EnqueueCommand(c, c.player, cmd, args)
	return nil
}
