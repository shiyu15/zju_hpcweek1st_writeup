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

package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strings"

	"go.uber.org/zap"

	"github.com/mrhaoxx/go-mc/chat"
	"github.com/mrhaoxx/go-mc/game"
	"github.com/mrhaoxx/go-mc/hpcworld"
	"github.com/mrhaoxx/go-mc/judge"
	"github.com/mrhaoxx/go-mc/server"
	"github.com/mrhaoxx/go-mc/world"

	_ "net/http/pprof"
)

func main() {
	flag.Parse()

	var (
		logger *zap.Logger
		err    error
	)

	logger, err = zap.NewProduction()

	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
		os.Exit(1)
	}

	// go func() {
	// 	log.Println(http.ListenAndServe("localhost:6060", nil))
	// }()

	exitCode := runApplication(logger)

	if exitCode != 0 {
		os.Exit(exitCode)
	}
}

func runApplication(logger *zap.Logger) int {
	args := flag.Args()
	if len(args) == 0 {
		config, err := readConfig()
		if err != nil {
			logger.Error("Read config fail", zap.Error(err))
			return 1
		}
		return runServerMode(logger, config)
	}

	switch args[0] {
	case "run":
		if len(args) < 2 {
			logger.Error("run mode requires a script filename")
			return 1
		}
		config, err := readConfig()
		if err != nil {
			logger.Error("Read config fail", zap.Error(err))
			return 1
		}
		var tpsPath string
		if len(args) >= 3 {
			tpsPath = args[2]
		}
		return runScriptMode(logger, config, args[1], tpsPath)
	case "compare":
		if len(args) != 3 {
			logger.Error("compare mode requires two .mccs files: compare <file1> <file2>")
			return 1
		}
		return runCompareMode(logger, args[1], args[2])
	default:
		logger.Error("unknown command", zap.String("command", args[0]))
		return 1
	}
}

func runServerMode(logger *zap.Logger, config game.Config) int {
	logger.Info("Server start")
	printBuildInfo(logger)
	defer logger.Info("Server exit")

	playerList := server.NewPlayerList(config.MaxPlayers)
	serverInfo := server.NewPingInfo(
		"Go-MC "+server.ProtocolName,
		server.ProtocolVersion,
		chat.Text(config.MessageOfTheDay),
		nil,
	)

	gp := game.NewGame(logger, config, playerList, serverInfo)
	startConsoleCommandBridge(gp.World(), logger)

	s := server.Server{
		Logger: zap.NewStdLog(logger),
		ListPingHandler: struct {
			*server.PlayerList
			*server.PingInfo
		}{playerList, serverInfo},
		LoginHandler: &server.MojangLoginHandler{
			OnlineMode:           false,
			EnforceSecureProfile: config.EnforceSecureProfile,
			Threshold:            config.NetworkCompressionThreshold,
			LoginChecker:         playerList,
		},
		ConfigHandler: &server.Configurations{
			Registries: world.NetworkCodec,
		},
		GamePlay: gp,
	}
	logger.Info("Start listening", zap.String("address", config.ListenAddress))
	if err := s.Listen(config.ListenAddress); err != nil {
		logger.Error("Server listening error", zap.Error(err))
		return 1
	}
	return 0
}

func runScriptMode(logger *zap.Logger, config game.Config, scriptPath, tpsPath string) int {
	scriptLogger := logger.Named("run")
	scriptLogger.Info("Starting script execution", zap.String("file", scriptPath))
	w := world.New(
		scriptLogger.Named("world"),
		world.Config{
			ViewDistance:  config.ViewDistance,
			SpawnPosition: [3]int32{0, 100, 0},
		},
	)
	w.SetJudgeProvider(judge.NewProvider())

	steps, err := w.RunScriptFile(scriptPath)
	if err != nil {
		scriptLogger.Error("Failed to execute script", zap.Error(err))
		return 1
	}

	for _, step := range steps {
		if step.Err != nil {
			scriptLogger.Warn("Unknown command", zap.Int("line", step.Line), zap.String("command", step.Command), zap.Error(step.Err))
		}
		if len(step.Result.Messages) == 0 {
			scriptLogger.Debug("Command executed", zap.Int("line", step.Line), zap.String("command", step.Command))
			continue
		}
		for _, msg := range step.Result.Messages {
			scriptLogger.Info("Command output", zap.Int("line", step.Line), zap.String("command", step.Command), zap.String("message", strings.TrimSpace(msg.Text)))
		}
	}

	scriptLogger.Info("Script execution complete", zap.String("file", scriptPath))
	return 0
}

func runCompareMode(logger *zap.Logger, fileA, fileB string) int {
	logger.Info("Comparing sequence files", zap.String("fileA", fileA), zap.String("fileB", fileB))
	equal, reason, err := compareMCCSFiles(fileA, fileB)
	if err != nil {
		logger.Error("Comparison failed", zap.Error(err))
		return 2
	}
	if equal {
		logger.Info("Sequence files match", zap.String("fileA", fileA), zap.String("fileB", fileB))
		return 0
	}
	logger.Warn("Sequence files differ", zap.String("fileA", fileA), zap.String("fileB", fileB), zap.String("reason", reason))
	return 1
}

func compareMCCSFiles(pathA, pathB string) (bool, string, error) {
	f1, err := os.Open(pathA)
	if err != nil {
		return false, "", err
	}
	defer f1.Close()

	chunkA, err := judge.NewDimStreamReader(f1)
	if err != nil {
		return false, "", err
	}

	f2, err := os.Open(pathB)
	if err != nil {
		return false, "", err
	}
	defer f2.Close()

	chunkB, err := judge.NewDimStreamReader(f2)
	if err != nil {
		return false, "", err
	}

	if chunkA.SampleRate() != chunkB.SampleRate() {
		return false, fmt.Sprintf("sample rate mismatch: %d vs %d", chunkA.SampleRate(), chunkB.SampleRate()), nil
	}
	if chunkA.Total() != chunkB.Total() {
		return false, fmt.Sprintf("tick count mismatch: %d vs %d", chunkA.Total(), chunkB.Total()), nil
	}

	for tick := 0; tick < chunkA.Total(); tick++ {
		if tick%chunkA.SampleRate() != 0 {
			continue
		}

		dimA, ok, err := chunkA.Next()
		if !ok || err != nil {
			return false, fmt.Sprintf("failed to read A dimension at tick %d: %v", tick, err), nil
		}
		dimB, ok, err := chunkB.Next()
		if !ok || err != nil {
			return false, fmt.Sprintf("failed to read B dimension at tick %d: %v", tick, err), nil
		}
		if len(dimA.Chunks) != len(dimB.Chunks) {
			return false, fmt.Sprintf("chunk count mismatch at tick %d: %d vs %d", tick, len(dimA.Chunks), len(dimB.Chunks)), nil
		}
		for idx := range dimA.Chunks {
			if !hpcworld.CompareChunk(&dimA.Chunks[idx], &dimB.Chunks[idx]) {
				return false, fmt.Sprintf("chunk mismatch at tick %d index %d", tick, idx), nil
			}
		}
		fmt.Println("Compared tick", tick)
	}

	return true, "", nil
}

// printBuildInfo reading compile information of the binary program with runtime/debug package，and print it to log
func printBuildInfo(logger *zap.Logger) {
	binaryInfo, _ := debug.ReadBuildInfo()
	settings := make(map[string]string)
	for _, v := range binaryInfo.Settings {
		settings[v.Key] = v.Value
	}
	logger.Info("Build info", zap.Any("settings", settings))
}

func readConfig() (game.Config, error) {
	var c game.Config = game.Config{
		ListenAddress:   ":25565",
		MaxPlayers:      20,
		ViewDistance:    10,
		MessageOfTheDay: "Not A Minecraft Server",
		OnlineMode:      false,
	}
	read_file, err := os.ReadFile("config.json")
	if err != nil && !os.IsNotExist(err) {
		return game.Config{}, err
	}
	if len(read_file) == 0 {
		// empty config file, use default config
		return c, nil
	}
	err = json.Unmarshal(read_file, &c)
	if err != nil {
		return game.Config{}, err
	}
	return c, nil
}

func startConsoleCommandBridge(w *world.World, logger *zap.Logger) {
	if w == nil {
		return
	}
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "/") {
				line = strings.TrimSpace(line[1:])
			}
			fields := strings.Fields(line)
			if len(fields) == 0 {
				w.EnqueueCommand(nil, nil, "", nil)
				continue
			}
			cmd := fields[0]
			args := append([]string(nil), fields[1:]...)
			w.EnqueueCommand(nil, nil, cmd, args)
		}
		if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
			logger.Warn("stdin command bridge stopped", zap.Error(err))
		}
	}()
}
