package client

import (
	"fmt"

	"github.com/mrhaoxx/go-mc/hpcworld"
	"github.com/mrhaoxx/go-mc/level/block"
	"github.com/mrhaoxx/go-mc/level/component"
	pk "github.com/mrhaoxx/go-mc/net/packet"
	"github.com/mrhaoxx/go-mc/world"
	"go.uber.org/zap"
)

type ItemStack struct {
	ItemID     int32
	Count      byte
	Components []component.DataComponent
}

func (s *ItemStack) isEmpty() bool {
	if s == nil {
		return true
	}
	if s.ItemID == 0 {
		return true
	}
	if s.Count == 0 {
		return true
	}
	return false
}

func (s *ItemStack) encodeFields() []pk.FieldEncoder {
	if s == nil || s.isEmpty() {
		return []pk.FieldEncoder{pk.VarInt(0)}
	}

	fields := []pk.FieldEncoder{
		pk.VarInt(s.Count), // non-empty
		pk.VarInt(s.ItemID),
		pk.VarInt(0),
		pk.VarInt(0), // No NBT
	}

	// fields = append(fields, encodeComponentPatch(s.Components)...)
	return fields
}

// clientUseItemOn handles right-click block placement.
func clientUseItemOn(p pk.Packet, c *Client) error {
	var (
		hand       pk.VarInt
		pos        pk.Position
		face       pk.VarInt
		fx, fy, fz pk.Float
		inside     pk.Boolean
		seq        pk.VarInt
	)
	if err := p.Scan(&hand, &pos, &face, &fx, &fy, &fz, &inside, &seq); err != nil {
		if err2 := p.Scan(&hand, &pos, &face, &fx, &fy, &fz, &inside); err2 != nil {
			return err
		}
	}

	fmt.Println("Client: UseItemOn", hand, pos, face, fx, fy, fz, inside, seq)
	// Compute placement position based on clicked face
	x, y, z := pos.X, pos.Y, pos.Z
	switch int(face) {
	case 0: // down
		y--
	case 1: // up
		y++
	case 2: // north (-z)
		z--
	case 3: // south (+z)
		z++
	case 4: // west (-x)
		x--
	case 5: // east (+x)
		x++
	}

	// Get the item from the player's carried slot
	if c.player.CarriedSlot >= 0 && c.player.CarriedSlot < 9 {
		itemStack := c.player.Inventory[c.player.CarriedSlot]
		if itemStack != nil {
			// Convert item ID to block state ID
			// For wool blocks, item ID = block state ID
			var blockStateID int32
			switch itemStack.ItemID {
			case 942:
				blockStateID = 86 // water
			case 943:
				blockStateID = 102 //lava
			case 829:
				blockStateID = 2403 // fire
			default:
				blockStateID = int32(block.ToStateID[itemIDToBlockState(itemStack.ItemID)])

			}
			fmt.Println("Placing block state ID", blockStateID, "for item ID", itemStack.ItemID)

			hpcworld.SetBlock(int32(x), int32(y), int32(z), blockStateID)
		}
	}
	return nil
}

// clientSetCarriedItem updates the player's selected hotbar slot.
func clientSetCarriedItem(p pk.Packet, c *Client) error {
	var slotS pk.Short
	if err := p.Scan(&slotS); err != nil {
		var slotV pk.VarInt
		if err2 := p.Scan(&slotV); err2 != nil {
			return err
		}
		slotS = pk.Short(slotV)
	}
	s := int32(slotS)
	if s < 0 {
		s = 0
	}
	if s > 8 {
		s = 8
	}
	c.player.CarriedSlot = s
	c.log.Info("Client: SetCarriedItem", zap.Int32("slot", s), zap.String("name", c.player.Name))
	return nil
}

// clientSetCreativeModeSlot handles creative inventory edits.
// Slot numbering follows vanilla (hotbar 36..44).
func clientSetCreativeModeSlot(p pk.Packet, c *Client) error {
	var slot pk.Short
	var count pk.VarInt

	if err := p.Scan(&slot, &count); err != nil {
		return err
	}

	idx := int32(slot)
	if count > 0 {
		var itemID pk.VarInt = 1
		var nAdd pk.VarInt
		var nDel pk.VarInt
		if err := p.Scan(&slot, &count, &itemID, &nAdd, &nDel); err != nil {
			return err
		}

		// Map slot index to inventory index
		var invIdx int32
		switch {
		case idx >= 36 && idx <= 44: // Hotbar
			invIdx = idx - 36
		case idx >= 9 && idx <= 35: // Main inventory
			invIdx = idx
		default:
			invIdx = idx
		}

		if invIdx >= 0 && invIdx < int32(len(c.player.Inventory)) {
			c.player.Inventory[invIdx] = &world.ItemStack{
				ItemID: int32(itemID),
				Count:  byte(count),
			}
		}

		// Update carried slot if player edited a hotbar index
		if idx >= 36 && idx <= 44 {
			c.player.CarriedSlot = idx - 36
		}
		c.log.Info("Client: SetCreativeModeSlot", zap.Int32("slot", idx), zap.Int32("itemID", int32(itemID)), zap.Int("nAdd", int(nAdd)), zap.Int("nDel", int(nDel)), zap.Int("count", int(count)), zap.String("name", c.player.Name))
	} else {
		// Empty slot
		var invIdx int32
		switch {
		case idx >= 36 && idx <= 44:
			invIdx = idx - 36
		case idx >= 9 && idx <= 35:
			invIdx = idx
		default:
			invIdx = idx
		}

		if invIdx >= 0 && invIdx < int32(len(c.player.Inventory)) {
			c.player.Inventory[invIdx] = nil
		}
		c.log.Info("Client: SetCreativeModeSlot (empty)", zap.Int32("slot", idx), zap.String("name", c.player.Name))
	}
	return nil
}
