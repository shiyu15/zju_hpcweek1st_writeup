package judge

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrhaoxx/go-mc/hpcworld"
	"github.com/mrhaoxx/go-mc/world"
)

type Judger struct {
	World    *world.World
	Ticks    int
	Interval int
	Filename string
	FileTps  string
	X1, Z1   int
	X2, Z2   int
}

type TimingResult struct {
	TotalTicks int     `json:"total_ticks"`
	TotalTime  float64 `json:"total_time_seconds"`
}

func (j *Judger) Create() error {
	if j.World.GetTickRate() != 0 {
		return errors.New("judge: world tick rate must be 0")
	}

	if j.Filename == "" {
		return errors.New("judge: empty filename")
	}

	file, err := os.Create(j.Filename)
	if err != nil {
		return err
	}
	defer file.Close()

	wr, err := NewDimStreamWriter(file, j.Interval, j.Ticks)
	if err != nil {
		return err
	}
	defer wr.Close()

	savings := []Dim{}

	starting := time.Now()

	for tick := 0; tick < j.Ticks; tick++ {
		if tick%j.Interval != 0 {
			j.World.Tick()
			continue
		}
		j.World.Tick()

		savings = append(savings, Dim{
			Chunks: make([]hpcworld.Chunk, (j.X2-j.X1)*(j.Z2-j.Z1)),
		})

		saving := &savings[len(savings)-1]

		for x := j.X1; x < j.X2; x++ {
			for z := j.Z1; z < j.Z2; z++ {
				chunk := j.World.GetChunk([2]int32{int32(x), int32(z)})
				if chunk == nil || chunk.Chunk == nil {
					saving.Chunks = append(saving.Chunks, hpcworld.Chunk{})
					fmt.Println("Warning: chunk", x, z, "is nil, saving empty chunk")
					continue
				}
				saving.Chunks[x-j.X1+(z-j.Z1)*(j.X2-j.X1)] = *chunk.Chunk
			}
		}
	}

	ending := time.Now()

	for _, saving := range savings {
		if err := wr.WriteDim(saving); err != nil {
			return err
		}
	}

	if err := wr.Close(); err != nil {
		return err
	}

	// Write timing result
	if j.FileTps != "" {
		tpsFile, err := os.Create(j.FileTps)
		if err != nil {
			return err
		}
		defer tpsFile.Close()

		timing := TimingResult{
			TotalTicks: j.Ticks,
			TotalTime:  ending.Sub(starting).Seconds(),
		}

		jsonEncoder := json.NewEncoder(tpsFile)
		if err := jsonEncoder.Encode(timing); err != nil {
			return err
		}
	}

	return nil
}

func (j *Judger) Compare() error {
	if j.World.GetTickRate() != 0 {
		return errors.New("judge: world tick rate must be 0")
	}

	if j.Filename == "" {
		return errors.New("judge: empty filename")
	}

	file, err := os.Open(j.Filename)
	if err != nil {
		return err
	}
	defer file.Close()

	wr, err := NewDimStreamReader(file)
	if err != nil {
		return err
	}
	defer wr.Close()

	if wr.SampleRate() != j.Interval {
		return fmt.Errorf("judge: interval mismatch, expected %d, got %d", j.Interval, wr.SampleRate())
	}
	if wr.Total() != j.Ticks {
		return fmt.Errorf("judge: ticks mismatch, expected %d, got %d", j.Ticks, wr.Total())
	}

	for tick := 0; tick < j.Ticks; tick++ {
		if tick%j.Interval != 0 {
			j.World.Tick()
			continue
		}
		j.World.Tick()

		dim, ok, err := wr.Next()
		if !ok {
			return err
		}

		for x := j.X1; x < j.X2; x++ {
			for z := j.Z1; z < j.Z2; z++ {
				chunk := j.World.GetChunk([2]int32{int32(x), int32(z)})
				if chunk == nil || chunk.Chunk == nil {
					fmt.Println("Warning: chunk", x, z, "is nil, skipping")
					continue
				}
				if !hpcworld.CompareChunk(chunk.Chunk, &dim.Chunks[x-j.X1+(z-j.Z1)*(j.X2-j.X1)]) {
					return fmt.Errorf("judge: chunk mismatch at (%d, %d) at tick %d", x, z, tick)
				}
			}
		}
	}
	return nil

}
