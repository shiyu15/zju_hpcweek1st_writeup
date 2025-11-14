package judge

import "github.com/mrhaoxx/go-mc/world"

type provider struct{}

// NewProvider returns a JudgeProvider implementation backed by the judge package.
func NewProvider() world.JudgeProvider {
	return provider{}
}

func (provider) Create(w *world.World, filename string, ticks, interval int, x1, z1, x2, z2 int, tpsFile string) error {
	j := Judger{
		World:    w,
		Ticks:    ticks,
		Interval: interval,
		Filename: filename,
		X1:       x1,
		Z1:       z1,
		X2:       x2,
		Z2:       z2,
		FileTps:  tpsFile,
	}
	return j.Create()
}

func (provider) Compare(w *world.World, filename string, ticks, interval int, x1, z1, x2, z2 int) error {
	j := Judger{
		World:    w,
		Ticks:    ticks,
		Interval: interval,
		Filename: filename,
		X1:       x1,
		Z1:       z1,
		X2:       x2,
		Z2:       z2,
	}
	return j.Compare()
}
