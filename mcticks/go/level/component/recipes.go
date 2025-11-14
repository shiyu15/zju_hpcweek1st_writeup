package component

import (
	"io"

	"github.com/mrhaoxx/go-mc/nbt/dynbt"
	pk "github.com/mrhaoxx/go-mc/net/packet"
)

var _ DataComponent = (*Recipes)(nil)

type Recipes struct {
	Data dynbt.Value
}

// ID implements DataComponent.
func (Recipes) ID() string {
	return "minecraft:recipes"
}

// ReadFrom implements DataComponent.
func (r *Recipes) ReadFrom(reader io.Reader) (n int64, err error) {
	return pk.NBT(&r.Data).ReadFrom(reader)
}

// WriteTo implements DataComponent.
func (r *Recipes) WriteTo(writer io.Writer) (n int64, err error) {
	return pk.NBT(&r.Data).WriteTo(writer)
}
