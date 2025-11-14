package world

import (
	"fmt"
	"os"
	"strings"

	"github.com/mrhaoxx/go-mc/chat"
)

// ScriptStep captures the outcome of executing a single line in a script file.
type ScriptStep struct {
	Line    int
	Command string
	Args    []string
	Result  CommandResult
	Err     error
}

// RunScriptFile executes the commands contained in the specified script file and
// returns the outcome for each processed line. Lines that are empty or start
// with a comment character (#) are ignored. Unknown commands are reported in the
// returned ScriptStep slice but do not interrupt execution.
func (w *World) RunScriptFile(filename string) ([]ScriptStep, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var steps []ScriptStep
	lines := strings.Split(string(data), "\n")
	for idx, raw := range lines {
		lineNumber := idx + 1
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		cmdName := parts[0]
		cmdArgs := append([]string(nil), parts[1:]...)

		step := ScriptStep{
			Line:    lineNumber,
			Command: cmdName,
			Args:    cmdArgs,
		}

		handler, ok := w.commands[cmdName]
		if !ok {
			step.Err = fmt.Errorf("unknown command: %s", cmdName)
			step.Result = CommandResult{Messages: []chat.Message{{Text: fmt.Sprintf("Unknown command: %s", cmdName)}}}
			steps = append(steps, step)
			continue
		}

		step.Result = handler(w, nil, nil, cmdArgs)
		steps = append(steps, step)
	}

	return steps, nil
}
