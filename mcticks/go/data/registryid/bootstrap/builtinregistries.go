package bootstrap

import (
	"github.com/mrhaoxx/go-mc/data/registryid"
	"github.com/mrhaoxx/go-mc/level/block"
	"github.com/mrhaoxx/go-mc/registry"
)

func RegisterBlocks(reg *registry.Registry[block.Block]) {
	reg.Clear()
	for i, key := range registryid.Block {
		id, val := reg.Put(key, block.FromID[key])
		if int32(i) != id || val == nil || *val == nil {
			panic("register blocks failed")
		}
	}
}
