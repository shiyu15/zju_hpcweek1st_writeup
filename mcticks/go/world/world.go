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

package world

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"

	"github.com/mrhaoxx/go-mc/animate"
	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/hpcworld"
	"github.com/mrhaoxx/go-mc/level"
	"github.com/mrhaoxx/go-mc/level/block"
	"github.com/mrhaoxx/go-mc/save"
	"github.com/mrhaoxx/go-mc/save/region"
	"github.com/mrhaoxx/go-mc/world/internal/bvh"
)

type World struct {
	log    *zap.Logger
	config Config

	chunks  map[[2]int32]*LoadedChunk
	loaders map[ChunkViewer]*loader

	tickLock sync.Mutex

	tickCount uint
	tickRate  uint
	// playerViews is a BVH tree，storing the visual range collision boxes of each player.
	// the data structure is used to determine quickly which players to send notify when entity moves.
	playerViews playerViewTree
	players     map[Client]*Player

	commandQueue  chan commandTask
	commands      map[string]CommandHandler
	judgeProvider JudgeProvider

	scheduledTemplates map[uint][]scheduledCommand

	TPS TPSSummary
}

type TPSSummary struct {
	TotalTicks    uint
	StartWallTime time.Time

	HPCTickTime time.Duration
	AnimateTime time.Duration
}

type Config struct {
	ViewDistance  int32
	SpawnAngle    float32
	SpawnPosition [3]int32
}

type playerView struct {
	EntityViewer
	*Player
}

type (
	vec3d          = bvh.Vec3[float64]
	aabb3d         = bvh.AABB[float64, vec3d]
	playerViewNode = bvh.Node[float64, aabb3d, playerView]
	playerViewTree = bvh.Tree[float64, aabb3d, playerView]
)

func New(logger *zap.Logger, config Config) (w *World) {
	w = &World{
		log:                logger,
		config:             config,
		chunks:             make(map[[2]int32]*LoadedChunk),
		loaders:            make(map[ChunkViewer]*loader),
		players:            make(map[Client]*Player),
		tickRate:           0,
		commandQueue:       make(chan commandTask, 128),
		commands:           make(map[string]CommandHandler),
		scheduledTemplates: make(map[uint][]scheduledCommand),
		TPS: TPSSummary{
			StartWallTime: time.Now(),
		},
	}

	// animate.SetBlockHandler(func(x, y, z, blockState int32) {
	// 	w.DirectSetBlock(x, y, z, level.BlocksState(blockState))
	// })

	w.registerDefaultCommands()

	go w.tickLoop()
	go w.chunkLoop()
	go w.commandLoop()
	return
}

type CommandResult struct {
	Messages []chat.Message
}

type CommandHandler func(*World, Client, *Player, []string) CommandResult

type JudgeProvider interface {
	Create(w *World, filename string, ticks, interval int, x1, z1, x2, z2 int, tpsFile string) error
	Compare(w *World, filename string, ticks, interval int, x1, z1, x2, z2 int) error
}

func (w *World) SetJudgeProvider(provider JudgeProvider) {
	w.judgeProvider = provider
}

type commandTask struct {
	source Client
	player *Player
	name   string
	args   []string
}

type scheduledCommand struct {
	name string
	args []string
}

func (w *World) EnqueueCommand(source Client, player *Player, name string, args []string) {
	taskArgs := append([]string(nil), args...)
	w.commandQueue <- commandTask{
		source: source,
		player: player,
		name:   name,
		args:   taskArgs,
	}
}

func (w *World) commandLoop() {
	for task := range w.commandQueue {
		player := task.player
		if player == nil {
			player = w.players[task.source]
		}
		sourceName := "Unknown"
		if player != nil {
			sourceName = player.Name
		} else if task.source == nil {
			sourceName = "Console"
		}

		// if task.name != "" {
		// 	w.broadcastSystemMessage(chat.Message{Text: w.formatCommandHeader(sourceName, task.name, task.args)})
		// }

		var result CommandResult
		if task.name == "" {
			result = CommandResult{Messages: []chat.Message{{Text: "Commands Separated by Space"}}}
		} else if handler, ok := w.commands[task.name]; ok {
			result = handler(w, task.source, player, task.args)
		} else {
			result = CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Unknown command: %s", task.name)}}}
		}
		w.broadcastCommandResult(sourceName, result)
	}
}

// func (w *World) formatCommandHeader(sourceName, name string, args []string) string {
// 	joinedArgs := strings.Join(args, " ")
// 	commandWord := name
// 	if commandWord == "" {
// 		commandWord = "<empty>"
// 	}
// 	if joinedArgs != "" {
// 		return fmt.Sprintf("[%s] /%s %s", sourceName, commandWord, joinedArgs)
// 	}
// 	return fmt.Sprintf("[%s] /%s", sourceName, commandWord)
// }

func (w *World) broadcastCommandResult(sourceName string, result CommandResult) {
	for _, msg := range result.Messages {
		message := msg
		trimmed := strings.TrimSpace(message.Text)
		if trimmed != "" {
			message.Text = fmt.Sprintf("[%s] %s", sourceName, trimmed)
			fmt.Println(message.Text)
		}
		w.broadcastSystemMessage(message)
	}
}

func (w *World) broadcastSystemMessage(msg chat.Message) {
	for c := range w.players {
		c.SendSystemChat(msg, false)
	}
}

type commandResponse struct {
	messages []chat.Message
}

func newCommandResponse() *commandResponse {
	return &commandResponse{messages: make([]chat.Message, 0)}
}

func (r *commandResponse) add(text string) {
	r.messages = append(r.messages, chat.Message{Text: text})
}

func (r *commandResponse) result() CommandResult {
	return CommandResult{Messages: append([]chat.Message(nil), r.messages...)}
}

func (w *World) RegisterCommand(name string, handler CommandHandler) {
	w.commands[name] = handler
}

func (w *World) registerDefaultCommands() {
	w.RegisterCommand("ping", commandPing)
	w.RegisterCommand("tp", commandTeleport)
	w.RegisterCommand("setblock", commandSetBlock)
	w.RegisterCommand("fill", commandFill)
	w.RegisterCommand("tick", commandTick)
	w.RegisterCommand("reload", commandReload)
	w.RegisterCommand("rate", commandRate)
	w.RegisterCommand("bench", commandBench)
	w.RegisterCommand("animate", commandAnimate)
	w.RegisterCommand("add_chunk", commandAddChunk)
	w.RegisterCommand("ensure_world", commandEnsureWorld)
	w.RegisterCommand("clear_world", commandClearWorld)
	w.RegisterCommand("clear_animate", commandClearAnimate)
	w.RegisterCommand("judge_create", commandJudgeCreate)
	w.RegisterCommand("judge_compare", commandJudgeCompare)
	w.RegisterCommand("script", commandRunScript)
	w.RegisterCommand("sched", commandSched)
	w.RegisterCommand("clear_sched", commandClearSched)
	w.RegisterCommand("reset_tick", commandResetTick)
	w.RegisterCommand("print_tick", commandPrintTick)
	w.RegisterCommand("settime", commandSetTime)
	w.RegisterCommand("clear_ticks", commandClearTicks)
	w.RegisterCommand("clear_tps", commandClearTPSummary)
	w.RegisterCommand("tps", commandTPS)
}

func commandPing(_ *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	res.add("Pong!")
	return res.result()
}

func commandTeleport(_ *World, source Client, player *Player, args []string) CommandResult {
	res := newCommandResponse()
	if player == nil || source == nil {
		res.add("Teleport requires a player context")
		return res.result()
	}
	if len(args) != 3 {
		res.add("Usage: /tp <x> <y> <z>")
		return res.result()
	}
	x, err1 := strconv.Atoi(args[0])
	y, err2 := strconv.Atoi(args[1])
	z, err3 := strconv.Atoi(args[2])
	if err1 != nil || err2 != nil || err3 != nil {
		res.add("Invalid coordinates")
		return res.result()
	}
	player.Position[0] = float64(x)
	player.Position[1] = float64(y)
	player.Position[2] = float64(z)
	source.SendPlayerPosition(player.Position, player.Rotation)
	res.add(fmt.Sprintf("Teleporting to %d %d %d", x, y, z))
	return res.result()
}

func commandSetBlock(_ *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 4 {
		res.add("Usage: /setblock <x> <y> <z> <block>")
		return res.result()
	}
	x, err1 := strconv.Atoi(args[0])
	y, err2 := strconv.Atoi(args[1])
	z, err3 := strconv.Atoi(args[2])
	blockID, err4 := strconv.Atoi(args[3])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	hpcworld.SetBlock(int32(x), int32(y), int32(z), int32(blockID))
	res.add(fmt.Sprintf("Setting block to %d at %d %d %d", blockID, x, y, z))
	return res.result()
}

func commandFill(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 7 {
		res.add("Usage: /fill <x1> <y1> <z1> <x2> <y2> <z2> <block>")
		return res.result()
	}
	x1, err1 := strconv.Atoi(args[0])
	y1, err2 := strconv.Atoi(args[1])
	z1, err3 := strconv.Atoi(args[2])
	x2, err4 := strconv.Atoi(args[3])
	y2, err5 := strconv.Atoi(args[4])
	z2, err6 := strconv.Atoi(args[5])
	blockID, err7 := strconv.Atoi(args[6])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil || err6 != nil || err7 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	if z1 > z2 {
		z1, z2 = z2, z1
	}
	volume := (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
	res.add(fmt.Sprintf("Filling %d blocks from (%d,%d,%d) to (%d,%d,%d) with block %d", volume, x1, y1, z1, x2, y2, z2, blockID))
	for x := x1; x <= x2; x++ {
		for y := y1; y <= y2; y++ {
			for z := z1; z <= z2; z++ {
				ck := w.GetChunk([2]int32{int32(x / 16), int32(z / 16)})
				if ck == nil {
					res.add(fmt.Sprintf("Chunk not found at (%d,%d)", x, z))
					continue
				}
				hpcworld.SetBlock(int32(x), int32(y), int32(z), int32(blockID))
			}
		}
	}
	res.add(fmt.Sprintf("Successfully filled %d blocks", volume))
	return res.result()
}

func commandTick(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()

	w.Tick()
	res.add(fmt.Sprintf("World ticked %d", w.GetTickCount()))
	return res.result()
}

func commandResetTick(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()

	w.tickLock.Lock()
	w.tickCount = 0
	w.tickLock.Unlock()

	res.add("World tick count reset to 0")
	return res.result()
}

func commandClearTicks(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()

	hpcworld.ClearTicks()
	res.add("Cleared all scheduled ticks")
	return res.result()
}

func commandClearTPSummary(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()

	w.TPS = TPSSummary{
		StartWallTime: time.Now(),
	}
	res.add("Cleared TPS summary")
	return res.result()
}

func commandTPS(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	currentDuration := time.Since(w.TPS.StartWallTime)
	res.add(fmt.Sprintf("Avg TPS: %.4f, Total ticks: %d, Total time: %s", float64(w.TPS.TotalTicks)/currentDuration.Seconds(), w.TPS.TotalTicks, currentDuration.String()))
	res.add(fmt.Sprintf(" - HPC     TPS: %.4f", float64(w.TPS.TotalTicks)/w.TPS.HPCTickTime.Seconds()))
	res.add(fmt.Sprintf(" - Animate TPS: %.4f", float64(w.TPS.TotalTicks)/w.TPS.AnimateTime.Seconds()))
	return res.result()
}

func commandPrintTick(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()

	w.tickLock.Lock()
	currentTick := w.tickCount
	w.tickLock.Unlock()

	res.add(fmt.Sprintf("Current world tick count: %d", currentTick))
	return res.result()
}

func commandReload(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	w.ReloadChunks()
	res.add("Reloaded all chunks")
	return res.result()
}

func commandSetTime(w *World, c Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 3 {
		res.add("Usage: /settime <world age> <time of day> <do daylight cycle>")
		return res.result()
	}
	worldAge, err1 := strconv.ParseInt(args[0], 10, 64)
	timeOfDay, err2 := strconv.ParseInt(args[1], 10, 64)
	doDaylightCycle := args[2] == "true"
	if err1 != nil || err2 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	for p := range w.players {
		p.SendSetTime(worldAge, timeOfDay, doDaylightCycle)
	}
	res.add(fmt.Sprintf("Set world time to %d %d", worldAge, timeOfDay))
	return res.result()
}

func commandRate(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 1 {
		res.add("Usage: /rate <rate>")
		return res.result()
	}
	rateValue, err := strconv.ParseUint(args[0], 10, 32)
	if err != nil {
		res.add("Invalid rate value")
		return res.result()
	}
	w.SetTickRate(uint(rateValue))
	res.add(fmt.Sprintf("Set tick rate to %d", rateValue))
	return res.result()
}

func commandBench(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 1 {
		res.add("Usage: /bench <ticks>")
		return res.result()
	}
	ticks, err := strconv.Atoi(args[0])
	if err != nil || ticks <= 0 {
		res.add("Ticks must be positive")
		return res.result()
	}
	if w.GetTickRate() != 0 {
		res.add("Set tick rate to 0 before benchmarking")
		return res.result()
	}
	res.add(fmt.Sprintf("Benchmarking %d ticks...", ticks))
	start := time.Now()
	for i := 0; i < ticks; i++ {
		w.Tick()
	}
	elapsed := time.Since(start)
	res.add(fmt.Sprintf("Benchmark complete: %d ticks in %s (%.2f TPS)", ticks, elapsed.String(), float64(ticks)/elapsed.Seconds()))
	return res.result()
}

func commandAddChunk(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 7 {
		res.add("Usage: /add_chunk <filename> <x_1> <z_1> <x_2> <z_2> <offset_x> <offset_z>")
		return res.result()
	}
	filename := args[0]
	x1, err1 := strconv.Atoi(args[1])
	z1, err2 := strconv.Atoi(args[2])
	x2, err3 := strconv.Atoi(args[3])
	z2, err4 := strconv.Atoi(args[4])
	offsetX, err5 := strconv.Atoi(args[5])
	offsetZ, err6 := strconv.Atoi(args[6])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil || err6 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	r, err := region.Open(filename)
	if err != nil {
		res.add(fmt.Sprintf("Failed to open region file %s: %v", filename, err))
		return res.result()
	}
	defer r.Close()

	for x := x1; x <= x2; x++ {
		for z := z1; z <= z2; z++ {
			if !r.ExistSector(x, z) {
				w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Chunk (%d, %d) does not exist in region file %s", x, z, filename)}}})
				continue
			}
			data, err := r.ReadSector(x, z)
			if err != nil {
				w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Failed to read chunk (%d, %d) from region file %s: %v", x, z, filename, err)}}})
				continue
			}
			var ck save.Chunk
			if err := ck.Load(data); err != nil {
				w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Failed to load chunk data from region file %s: %v", filename, err)}}})
				continue
			}
			ckc, err := level.ChunkFromSave(&ck)
			if err != nil {
				w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Failed to convert chunk data from region file %s: %v", filename, err)}}})
				continue
			}
			chunk := w.GetChunk([2]int32{int32(x + offsetX), int32(z + offsetZ)})
			if chunk == nil {
				w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Failed to get chunk at (%d,%d)", x+offsetX, z+offsetZ)}}})
				continue
			}

			for i, s := range ckc.Sections {
				for j := 0; j < 16*16*16; j++ {
					b := s.GetBlock(j)
					chunk.Sections[i].BlocksState[j] = int32(b)
					chunk.Sections[i].SkyLight[j/2] = 127
				}
			}
			chunk.Chunk.LastUpdate = -1
			w.broadcastCommandResult("add_chunk", CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Applied chunk (%d,%d) from %s to (%d,%d)", x, z, filename, x+offsetX, z+offsetZ)}}})
		}
	}

	return res.result()
}

func commandEnsureWorld(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 4 {
		res.add("Usage: /ensure_world <x1> <z1> <x2> <z2>")
		return res.result()
	}
	x1, err1 := strconv.Atoi(args[0])
	z1, err2 := strconv.Atoi(args[1])
	x2, err3 := strconv.Atoi(args[2])
	z2, err4 := strconv.Atoi(args[3])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if z1 > z2 {
		z1, z2 = z2, z1
	}

	w.tickLock.Lock()

	for x := x1; x <= x2; x += 1 {
		for z := z1; z <= z2; z += 1 {
			if _, ok := w.chunks[[2]int32{int32(x), int32(z)}]; ok {
				continue
			}
			if !w.loadChunk([2]int32{int32(x), int32(z)}) {
				res.add(fmt.Sprintf("Failed to load chunk at (%d, %d)", x, z))
			}
		}
	}
	w.tickLock.Unlock()
	res.add(fmt.Sprintf("Ensured world from (%d,%d) to (%d,%d)", x1, z1, x2, z2))
	return res.result()
}

func commandClearWorld(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	w.Clear()
	res.add("World cleared")
	return res.result()
}

func commandClearAnimate(_ *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	animate.Clear()
	res.add("Animations cleared")
	return res.result()
}

func commandJudgeCreate(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) < 7 {
		res.add("Usage: /judge_create <filename> <ticks> <interval> <x1> <z1> <x2> <z2> [tps_file]")
		return res.result()
	}
	if w.judgeProvider == nil {
		res.add("Judge service unavailable")
		return res.result()
	}
	filename := args[0]
	ticks, err1 := strconv.Atoi(args[1])
	interval, err2 := strconv.Atoi(args[2])
	x1, err3 := strconv.Atoi(args[3])
	z1, err4 := strconv.Atoi(args[4])
	x2, err5 := strconv.Atoi(args[5])
	z2, err6 := strconv.Atoi(args[6])
	tps := ""
	if len(args) == 8 {
		tps = args[7]
	}
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil || err6 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	if ticks <= 0 || interval <= 0 {
		res.add("Ticks and interval must be positive")
		return res.result()
	}
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if z1 > z2 {
		z1, z2 = z2, z1
	}
	res.add(fmt.Sprintf("Judging %s for %d ticks with interval %d...", filename, ticks, interval))
	if err := w.judgeProvider.Create(w, filename, ticks, interval, x1, z1, x2, z2, tps); err != nil {
		res.add(fmt.Sprintf("Failed to create judge: %v", err))
		return res.result()
	}
	res.add("Judge complete")
	return res.result()
}

func commandJudgeCompare(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 7 {
		res.add("Usage: /judge_compare <filename> <ticks> <interval> <x1> <z1> <x2> <z2>")
		return res.result()
	}
	if w.judgeProvider == nil {
		res.add("Judge service unavailable")
		return res.result()
	}
	filename := args[0]
	ticks, err1 := strconv.Atoi(args[1])
	interval, err2 := strconv.Atoi(args[2])
	x1, err3 := strconv.Atoi(args[3])
	z1, err4 := strconv.Atoi(args[4])
	x2, err5 := strconv.Atoi(args[5])
	z2, err6 := strconv.Atoi(args[6])
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil || err6 != nil {
		res.add("Invalid arguments")
		return res.result()
	}
	if ticks <= 0 || interval <= 0 {
		res.add("Ticks and interval must be positive")
		return res.result()
	}
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if z1 > z2 {
		z1, z2 = z2, z1
	}
	res.add(fmt.Sprintf("Comparing %s for %d ticks with interval %d...", filename, ticks, interval))
	if err := w.judgeProvider.Compare(w, filename, ticks, interval, x1, z1, x2, z2); err != nil {
		res.add(fmt.Sprintf("Failed to compare judge: %v", err))
		return res.result()
	}
	res.add("Judge compare complete")
	return res.result()
}

func commandRunScript(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) != 1 {
		res.add("Usage: /run_script <filename>")
		return res.result()
	}
	filename := args[0]

	steps, err := w.RunScriptFile(filename)
	if err != nil {
		res.add(fmt.Sprintf("Failed to execute script %s: %v", filename, err))
		return res.result()
	}

	for _, step := range steps {
		if step.Err != nil {
			w.broadcastCommandResult("script: "+filename, CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("[Line %d] %s", step.Line, step.Err.Error())}}})
			continue
		}
		for _, msg := range step.Result.Messages {
			w.broadcastCommandResult("script: "+filename, CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("[Line %d] %s", step.Line, msg.Text)}}})
		}
	}

	res.add(fmt.Sprintf("Script %s execution complete", filename))
	return res.result()
}

func commandAnimate(_ *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) == 0 {
		res.add("Usage: /animate <create|move|spin|orbit> ...")
		return res.result()
	}
	msg, err := animate.Handle(args)
	if err != nil {
		res.add(err.Error())
		return res.result()
	}
	if msg != "" {
		res.add(msg)
	}
	return res.result()
}

func commandSched(w *World, _ Client, _ *Player, args []string) CommandResult {
	res := newCommandResponse()
	if len(args) < 2 {
		res.add("Usage: /sched <tick> <command> [args...]")
		return res.result()
	}
	tickValue, err := strconv.ParseUint(args[0], 10, 64)
	if err != nil {
		res.add("Invalid tick value")
		return res.result()
	}
	commandName := args[1]
	if _, ok := w.commands[commandName]; !ok {
		res.add(fmt.Sprintf("Unknown command: %s", commandName))
		return res.result()
	}
	cmdArgs := append([]string(nil), args[2:]...)
	w.addScheduledCommand(uint(tickValue), commandName, cmdArgs)
	res.add(fmt.Sprintf("Scheduled /%s at tick %d", commandName, tickValue))
	return res.result()
}

func (w *World) addScheduledCommand(tick uint, name string, args []string) {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()

	if w.scheduledTemplates == nil {
		w.scheduledTemplates = make(map[uint][]scheduledCommand)
	}
	entry := scheduledCommand{
		name: name,
		args: append([]string(nil), args...),
	}
	w.scheduledTemplates[tick] = append(w.scheduledTemplates[tick], entry)
}

func commandClearSched(w *World, _ Client, _ *Player, _ []string) CommandResult {
	res := newCommandResponse()
	w.tickLock.Lock()
	w.scheduledTemplates = make(map[uint][]scheduledCommand)
	w.tickLock.Unlock()
	res.add("Cleared all scheduled commands")
	return res.result()
}

func (w *World) tickScheduler(currentTick uint) {
	if len(w.scheduledTemplates) == 0 {
		return
	}
	cmds, ok := w.scheduledTemplates[currentTick]
	if !ok || len(cmds) == 0 {
		return
	}
	delete(w.scheduledTemplates, currentTick)
	result := make([]scheduledCommand, len(cmds))
	for i, cmd := range cmds {
		result[i] = scheduledCommand{
			name: cmd.name,
			args: append([]string(nil), cmd.args...),
		}
	}
	for _, sc := range result {
		handler, ok := w.commands[sc.name]
		if !ok {
			w.log.Warn("Scheduled command missing", zap.String("command", sc.name))
			continue
		}
		args := append([]string(nil), sc.args...)
		result := handler(w, nil, nil, args)
		w.broadcastCommandResult("scheduler", result)
	}
}

func (w *World) Name() string {
	return "minecraft:overworld"
}

func (w *World) SpawnPositionAndAngle() ([3]int32, float32) {
	return w.config.SpawnPosition, w.config.SpawnAngle
}

func (w *World) HashedSeed() [8]byte {
	return [8]byte{}
}

func (w *World) AddPlayer(c Client, p *Player, limiter *rate.Limiter) {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()
	w.loaders[c] = newLoader(p, limiter)
	w.players[c] = p
	p.view = w.playerViews.Insert(p.getView(), playerView{c, p})
}

func (w *World) RemovePlayer(c Client, p *Player) {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()
	w.log.Debug("Remove Player",
		zap.Int("loader count", len(w.loaders[c].loaded)),
		zap.Int("world count", len(w.chunks)),
	)
	// delete the player from all chunks which load the player.
	for pos := range w.loaders[c].loaded {
		if !w.chunks[pos].RemoveViewer(c) {
			w.log.Panic("viewer is not found in the loaded chunk")
		}
	}
	delete(w.loaders, c)
	delete(w.players, c)
	// delete the player from entity system.
	w.playerViews.Delete(p.view)
	w.playerViews.Find(
		bvh.TouchPoint[vec3d, aabb3d](bvh.Vec3[float64](p.Position)),
		func(n *playerViewNode) bool {
			n.Value.ViewRemoveEntities([]int32{p.EntityID})
			delete(n.Value.EntitiesInView, p.EntityID)
			return true
		},
	)
}

func (w *World) loadChunk(pos [2]int32) bool {
	logger := w.log.With(zap.Int32("x", pos[0]), zap.Int32("z", pos[1]))
	logger.Info("Loading chunk")

	c := hpcworld.LoadChunk(pos[0], pos[1])

	w.chunks[pos] = &LoadedChunk{Chunk: c, Pos: level.ChunkPos{pos[0], pos[1]}, Cvt: level.EmptyChunk(24)}
	return true
}

func (w *World) ReloadChunks() {
	for pos, ck := range w.chunks {
		ck.Mutex.Lock()
		ck.Chunk = hpcworld.LoadChunk(pos[0], pos[1])
		ck.Mutex.Unlock()
	}
}

func (w *World) Clear() {
	for _, ck := range w.chunks {
		ck.Mutex.Lock()
		for i := range ck.Chunk.Sections {
			ck.Chunk.Sections[i].BlockLight = [2048]byte{}
			ck.Chunk.Sections[i].SkyLight = [2048]byte{}
			ck.Chunk.Sections[i].BlocksState = [4096]int32{}
			ck.Chunk.Sections[i].Blockcount = 0
		}
		ck.Mutex.Unlock()
	}
}

// func (w *World) unloadChunk(pos [2]int32) {
// 	logger := w.log.With(zap.Int32("x", pos[0]), zap.Int32("z", pos[1]))
// 	logger.Debug("Unloading chunk")
// 	c, ok := w.chunks[pos]
// 	if !ok {
// 		logger.Panic("Unloading an non-exist chunk")
// 	}
// 	// notify all viewers who are watching the chunk to unload the chunk
// 	for _, viewer := range c.viewers {
// 		viewer.ViewChunkUnload(pos)
// 	}

// 	delete(w.chunks, pos)
// }

func (w *World) GetChunk(pos [2]int32) *LoadedChunk {
	return w.chunks[pos]
}

func (w *World) SetTickRate(tickRate uint) {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()
	w.tickRate = tickRate
}

func (w *World) GetTickRate() uint {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()
	return w.tickRate
}

func (w *World) ResetTickCount() {
	w.tickLock.Lock()
	defer w.tickLock.Unlock()
	w.tickCount = 0
}

func (w *World) BroadcastSwing(p *Player, animation byte) {
	cond := bvh.TouchPoint[vec3d, aabb3d](vec3d(p.Position))
	w.playerViews.Find(cond, func(n *playerViewNode) bool {
		if n.Value.Player == p {
			return true
		}
		n.Value.ViewAnimate(p.EntityID, animation)
		return true
	})
}

type LoadedChunk struct {
	sync.Mutex
	viewers []ChunkViewer
	*hpcworld.Chunk
	Cvt         *level.Chunk
	Lastrefresh int32
	Pos         level.ChunkPos
}

func (lc *LoadedChunk) refresh() {
	if lc.Lastrefresh == lc.Chunk.LastUpdate {
		return
	}
	lc.Lastrefresh = lc.Chunk.LastUpdate

	for j := range lc.Cvt.Sections {
		// for i := 0; i < 2048; i++ {
		// lc.Cvt.Sections[j].SkyLight[i] = byte(lc.Chunk.Sections[j].SkyLight[i])
		lc.Cvt.Sections[j].SkyLight = lc.Chunk.Sections[j].SkyLight[:]

		// }
		// for i := 0; i < 2048; i++ {
		// lc.Cvt.Sections[j].BlockLight[i] = byte(lc.Chunk.Sections[j].BlockLight[i])
		// }

		lc.Cvt.Sections[j].BlockLight = lc.Chunk.Sections[j].BlockLight[:]

		for i := 0; i < 16*16*16; i++ {
			// BlocksState in HPC is int32; level.BlocksState is alias of int
			if lc.Chunk.Sections[j].BlocksState[i] < 0 || lc.Chunk.Sections[j].BlocksState[i] >= int32(len(block.StateList)) {
				// Invalid block state ID; treat as air (0)
				fmt.Println("Warning: invalid block state ID", lc.Chunk.Sections[j].BlocksState[i], "at section", j, "index", i)
				lc.Cvt.Sections[j].States.Set(i, 0)
			} else {
				lc.Cvt.Sections[j].SetBlock(i, level.BlocksState(lc.Chunk.Sections[j].BlocksState[i]))
			}
		}
	}
}

func (lc *LoadedChunk) AddViewer(v ChunkViewer) {
	lc.Lock()
	defer lc.Unlock()
	for _, v2 := range lc.viewers {
		if v2 == v {
			panic("append an exist viewer")
		}
	}
	lc.viewers = append(lc.viewers, v)
}

func (lc *LoadedChunk) RemoveViewer(v ChunkViewer) bool {
	lc.Lock()
	defer lc.Unlock()
	for i, v2 := range lc.viewers {
		if v2 == v {
			last := len(lc.viewers) - 1
			lc.viewers[i] = lc.viewers[last]
			lc.viewers = lc.viewers[:last]
			return true
		}
	}
	return false
}

func (lc *LoadedChunk) SetBlock(x, y, z int, block level.BlocksState) {
	lc.Lock()
	defer lc.Unlock()
	y += 64
	// lc.Chunk.Sections[y/16].SetBlock((x%16)*16*16+(y%16)*16+z%16, block)

	// get all >0 values

	tx := x % 16
	if tx < 0 {
		tx += 16
	}
	tz := z % 16
	if tz < 0 {
		tz += 16
	}

	lc.Chunk.Sections[y/16].SetBlock((y%16)*16*16+(tz%16)*16+tx%16, int32(block))

}

func (lc *LoadedChunk) UpdateToViewers() {
	lc.Lock()
	defer lc.Unlock()
	if len(lc.viewers) == 0 {
		return
	}
	lc.refresh()

	for _, v := range lc.viewers {
		v.ViewChunkLoad(lc.Pos, lc)
	}

}

// updateRainbowInventory cycles rainbow colors through the top row of each player's inventory
func (w *World) updateRainbowInventory() {
	rainbowColors := []int32{
		209, // White Wool
		210, // Orange Wool
		211, // Magenta Wool
		212, // Light Blue Wool
		213, // Yellow Wool
		214, // Lime Wool
		215, // Pink Wool
		216, // Gray Wool
		217, // Light Gray Wool
		218, // Cyan Wool
		219, // Purple Wool
		220, // Blue Wool
		221, // Brown Wool
		222, // Green Wool
		223, // Red Wool
		224, // Black Wool
	}

	for c, p := range w.players {
		// Rotate colors for slots 9-17 (top inventory row)
		for i := 9; i < 10; i++ {
			colorIdx := (i - 9 + int(w.tickCount)) % len(rainbowColors)
			p.Inventory[i] = &ItemStack{
				ItemID: rainbowColors[colorIdx],
				Count:  1,
			}
			c.SendSetPlayerInventorySlot(int32(i), p.Inventory[i])
		}
	}
}

func (w *World) DirectSetBlock(x, y, z int32, block level.BlocksState) {
	ck := w.GetChunk([2]int32{x / 16, z / 16})
	if ck == nil {
		return
	}
	ck.SetBlock(int(x), int(y), int(z), block)
}
