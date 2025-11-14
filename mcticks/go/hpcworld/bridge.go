package hpcworld

/*
#cgo LDFLAGS: -lmc_tick_lib -L../../build
#include "bridge.h"

*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/mrhaoxx/go-mc/level"
	"github.com/mrhaoxx/go-mc/level/block"
)

// cgo: expose load_chunk
/*
Chunk* load_chunk(int32_t x, int32_t z);
*/

// LoadChunk wraps the C load_chunk function and returns a Go pointer to Chunk
func LoadChunk(x, z int32) *Chunk {
	cChunk := C.load_chunk(C.int32_t(x), C.int32_t(z))
	if cChunk == nil {
		return nil
	}
	return ToGoChunk(cChunk)
}

type Section struct {
	Blockcount int16

	BlocksState [4096]int32
	Biomes      [64]int32
	SkyLight    [2048]byte
	BlockLight  [2048]byte
}

type SetblockRequest struct {
	X          int32
	Y          int32
	Z          int32
	BlockState int32
}

// SetBlock sets the block at index i to v, updating Blockcount like level.Section.SetBlock
func (s *Section) SetBlock(i int, v int32) {
	// Air block is 0 in go-mc/level/block
	if s.BlocksState[i] != 0 {
		s.Blockcount--
	}
	if v != 0 {
		s.Blockcount++
	}
	s.BlocksState[i] = v
}

type Chunk struct {
	LastUpdate int32
	Sections   [24]Section
}

func CompareChunk(a, b *Chunk) bool {
	if a == nil || b == nil {
		return a == b
	}

	for i := 0; i < 24; i++ {
		asa := &a.Sections[i]
		asb := &b.Sections[i]
		if asa.Blockcount != asb.Blockcount {
			fmt.Println("Blockcount mismatch at section", i, ":", asa.Blockcount, "!=", asb.Blockcount)
			return false
		}
		for j := 0; j < 4096; j++ {
			if asa.BlocksState[j] != asb.BlocksState[j] {
				fmt.Println("BlockState mismatch at section", i, "index", j, ":", asa.BlocksState[j], "!=", asb.BlocksState[j])
				return false
			}
		}
		for j := 0; j < 2048; j++ {
			if asa.SkyLight[j] != asb.SkyLight[j] {
				fmt.Println("SkyLight mismatch at section", i, "index", j, ":", asa.SkyLight[j], "!=", asb.SkyLight[j])
				return false
			}
			if asa.BlockLight[j] != asb.BlockLight[j] {
				fmt.Println("BlockLight mismatch at section", i, "index", j, ":", asa.BlockLight[j], "!=", asb.BlockLight[j])
				return false
			}
		}
	}
	return true
}

func init() {
	var s Section
	if unsafe.Sizeof(s) != uintptr(C.Section_size) {
		panic("Section size mismatch")
	}
	if unsafe.Alignof(s) != uintptr(C.Section_align) {
		panic("Section align mismatch")
	}
	if unsafe.Offsetof(s.Blockcount) != uintptr(C.Section_off_blockcount) {
		panic("Section.blockcount off")
	}
	if unsafe.Offsetof(s.BlocksState) != uintptr(C.Section_off_blocks_state) {
		panic("Section.blocks_state off")
	}
	if unsafe.Offsetof(s.Biomes) != uintptr(C.Section_off_biomes) {
		panic("Section.biomes off")
	}
	if unsafe.Offsetof(s.SkyLight) != uintptr(C.Section_off_sky_light) {
		panic("Section.sky_light off")
	}
	if unsafe.Offsetof(s.BlockLight) != uintptr(C.Section_off_block_light) {
		panic("Section.block_light off")
	}

	var c Chunk
	if unsafe.Sizeof(c) != uintptr(C.Chunk_size) {
		panic("Chunk size mismatch")
	}
	if unsafe.Alignof(c) != uintptr(C.Chunk_align) {
		panic("Chunk align mismatch")
	}
	if unsafe.Offsetof(c.Sections) != uintptr(C.Chunk_off_sections) {
		panic("Chunk.sections off")
	}
}

func NewChunk() *C.Chunk    { return (*C.Chunk)(C.malloc(C.Chunk_size)) }
func FreeChunk(ch *C.Chunk) { C.free(unsafe.Pointer(ch)) }

// ToGoChunk converts a C Chunk pointer to a Go Chunk pointer for testing
func ToGoChunk(ch *C.Chunk) *Chunk {
	return (*Chunk)(unsafe.Pointer(ch))
}

// ToCChunk converts a Go Chunk pointer back to a C Chunk pointer
func ToCChunk(ch *Chunk) *C.Chunk {
	return (*C.Chunk)(unsafe.Pointer(ch))
}

// ToCSection converts a C Section pointer to a Go Section pointer for testing
func ToCSection(s *C.Section) *Section {
	return (*Section)(unsafe.Pointer(s))
}

// LevelChunkFromHPC converts an HPC C-backed Chunk into a high-level level.Chunk.
// It builds palette containers for block states and biomes and copies light data.
func LevelChunkFromHPC(ch *Chunk) *level.Chunk {
	if ch == nil {
		return level.EmptyChunk(24)
	}

	goCh := ch
	// HPC has a fixed-size array [24]Section; treat all as present
	secs := len(goCh.Sections)
	lc := level.EmptyChunk(secs)

	// Populate sections
	for si := 0; si < secs; si++ {
		hpcSec := &goCh.Sections[si]
		lvlSec := &lc.Sections[si]

		// States: create an empty palette container then set every block index
		lvlSec.States = level.NewStatesPaletteContainer(16*16*16, 0)
		for i := 0; i < 16*16*16; i++ {
			// BlocksState in HPC is int32; level.BlocksState is alias of int
			if hpcSec.BlocksState[i] < 0 || hpcSec.BlocksState[i] >= int32(len(block.StateList)) {
				// Invalid block state ID; treat as air (0)
				fmt.Println("Warning: invalid block state ID", hpcSec.BlocksState[i], "at section", si, "index", i)
				lvlSec.States.Set(i, 0)
			} else {
				lvlSec.SetBlock(i, level.BlocksState(hpcSec.BlocksState[i]))
			}
		}

		lvlSec.Biomes = level.NewBiomesPaletteContainer(4*4*4, 0)
		for i := 0; i < 4*4*4; i++ {
			lvlSec.Biomes.Set(i, level.BiomesState(0))
		}

		if lvlSec.SkyLight == nil || len(lvlSec.SkyLight) != 2048 {
			lvlSec.SkyLight = make([]byte, 2048)
		}
		for i := 0; i < 2048; i++ {
			lvlSec.SkyLight[i] = byte(hpcSec.SkyLight[i])
		}
		if lvlSec.BlockLight == nil || len(lvlSec.BlockLight) != 2048 {
			lvlSec.BlockLight = make([]byte, 2048)
		}
		for i := 0; i < 2048; i++ {
			lvlSec.BlockLight[i] = byte(hpcSec.BlockLight[i])
		}
	}

	// Status: no HPC notion; default to empty
	lc.Status = level.StatusEmpty

	return lc
}

func Tick() {
	C.tick_chunk()
}

func SetBlock(x, y, z int32, block int32) {
	C.setblock(C.int32_t(x), C.int32_t(y), C.int32_t(z), C.int32_t(block))
	C.tickAftersetblock()
}

func ClearTicks() {
	C.clear_ticks()
}

func Batch_SetBlocks(placements []SetblockRequest) {
	if len(placements) == 0 {
		return
	}
	C.batch_setblock(C.int(len(placements)), (*C.SetblockRequest)(unsafe.Pointer(&placements[0])))
	C.tickAftersetblock()
}
