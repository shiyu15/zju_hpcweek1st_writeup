package animate

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mrhaoxx/go-mc/hpcworld"
)

// ShapeType defines supported geometric primitives.
type ShapeType string

const (
	ShapeSphere ShapeType = "sphere"
	ShapeRect   ShapeType = "rect"
	ShapeRing   ShapeType = "ring"
	ShapeRandom ShapeType = "random"
)

// Manager stores named shapes and their animations.
type Manager struct {
	mu     sync.RWMutex
	shapes map[string]*Shape
	order  []string
	active map[[3]int32]int32
	diffs  []tickDiff

	SetblockTime time.Duration

	precomputed        bool
	precomputedMaxTick uint
	lastTick           uint
}

type tickDiff struct {
	clears []hpcworld.SetblockRequest
	fills  []hpcworld.SetblockRequest
}

// Shape describes a geometric object together with animation metadata.
type Shape struct {
	Name       string
	Type       ShapeType
	Filled     bool
	BlockState int32

	origin Vec3
	cells  []blockOffset

	lastBlocks []hpcworld.SetblockRequest

	Random *RandomSpec

	DestroyTick      uint
	DestroyScheduled bool

	Radius      float64
	InnerRadius float64
	Width       int
	Height      int

	Moves  []Movement
	Spins  []Spin
	Orbits []Orbit
}

// Movement configures linear motion for a shape.
type Movement struct {
	DX, DY, DZ float64
	StartTick  uint
	EndTick    uint
}

// Spin configures angular velocity (radians per tick) for a shape.
type Spin struct {
	RadX, RadY, RadZ float64
	StartTick        uint
	EndTick          uint
}

// Orbit configures circular motion within a plane.
type Orbit struct {
	Plane        OrbitPlane
	Radius       float64
	AngularSpeed float64
	Phase        float64
	StartTick    uint
	EndTick      uint
}

type RandomSpec struct {
	StartTick  uint
	EndTick    uint
	MinX       int32
	MaxX       int32
	MinY       int32
	MaxY       int32
	MinZ       int32
	MaxZ       int32
	Placements []hpcworld.SetblockRequest
}

// OrbitPlane represents the 2D plane of circular motion.
type OrbitPlane string

const (
	OrbitPlaneXY OrbitPlane = "xy"
	OrbitPlaneXZ OrbitPlane = "xz"
	OrbitPlaneYZ OrbitPlane = "yz"
)

// Vec3 represents a position or displacement in 3D space.
type Vec3 struct {
	X, Y, Z float64
}

type blockOffset struct {
	X, Y, Z int
}

// // hpcworld.SetblockRequest represents a concrete block update for the world or a chunk.
// type hpcworld.SetblockRequest struct {
// 	X, Y, Z    int32
// 	BlockState int32
// }

var (
	defaultManager = NewManager()
)

// NewManager constructs a Manager.
func NewManager() *Manager {
	return &Manager{shapes: make(map[string]*Shape)}
}

// Handle routes animate sub-commands using the default manager.
func Handle(args []string) (string, error) {
	return defaultManager.Handle(args)
}

// Tick advances all registered animations to the provided world tick.
func Tick(t uint) {
	defaultManager.Tick(t)
}

func PlacementCosts() time.Duration {
	return defaultManager.SetblockTime
}

// Clear removes all registered shapes and clears their blocks from the world.
func Clear() {
	defaultManager.Clear()
}

// Destory schedules a single named shape for destruction at the given tick.
// It returns true if the shape existed.
func Destory(name string, tick uint) bool {
	return defaultManager.Destory(name, tick)
}

// Destroy is an alias for Destory to provide the correct spelling.
func Destroy(name string, tick uint) bool {
	return defaultManager.Destory(name, tick)
}

// Handle routes animate sub-commands.
func (m *Manager) Handle(args []string) (string, error) {
	if len(args) == 0 {
		return "", errors.New("usage: animate <create|move|spin|orbit|destroy|precompute>")
	}
	cmd := strings.ToLower(args[0])
	switch cmd {
	case "create":
		return m.handleCreate(args[1:])
	case "move":
		return m.handleMove(args[1:])
	case "spin":
		return m.handleSpin(args[1:])
	case "orbit":
		return m.handleOrbit(args[1:])
	case "destory", "destroy":
		return m.handleDestory(args[1:])
	case "precompute":
		return m.handlePrecompute(args[1:])
	default:
		return "", fmt.Errorf("unknown animate action: %s", cmd)
	}
}

func (m *Manager) ensureMutable() error {
	m.mu.RLock()
	precomputed := m.precomputed
	m.mu.RUnlock()
	if precomputed {
		return errors.New("animate state already precomputed; no further modifications allowed")
	}
	return nil
}

// Shape retrieves a stored shape by name.
func (m *Manager) Shape(name string) (*Shape, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	sh, ok := m.shapes[name]
	return sh, ok
}

func (m *Manager) handleCreate(args []string) (string, error) {
	if len(args) < 2 {
		return "", errors.New("usage: animate create <name> <sphere|rect|ring|random>")
	}
	if err := m.ensureMutable(); err != nil {
		return "", err
	}
	name := args[0]
	shapeType := strings.ToLower(args[1])
	switch ShapeType(shapeType) {
	case ShapeSphere:
		return m.createSphere(name, args[2:])
	case ShapeRect:
		return m.createRect(name, args[2:])
	case ShapeRing:
		return m.createRing(name, args[2:])
	case ShapeRandom:
		return m.createRandom(name, args[2:])
	default:
		return "", fmt.Errorf("unsupported shape type: %s", shapeType)
	}
}

func (m *Manager) handleMove(args []string) (string, error) {
	if len(args) != 6 {
		return "", errors.New("usage: animate move <name> <dx> <dy> <dz> <start_tick> <end_tick>")
	}
	if err := m.ensureMutable(); err != nil {
		return "", err
	}
	name := args[0]
	dx, err := strconv.ParseFloat(args[1], 64)
	if err != nil {
		return "", fmt.Errorf("invalid dx: %w", err)
	}
	dy, err := strconv.ParseFloat(args[2], 64)
	if err != nil {
		return "", fmt.Errorf("invalid dy: %w", err)
	}
	dz, err := strconv.ParseFloat(args[3], 64)
	if err != nil {
		return "", fmt.Errorf("invalid dz: %w", err)
	}
	startTick, err := parseUint(args[4])
	if err != nil {
		return "", fmt.Errorf("invalid start tick: %w", err)
	}
	endTick, err := parseUint(args[5])
	if err != nil {
		return "", fmt.Errorf("invalid end tick: %w", err)
	}
	if endTick < startTick {
		return "", errors.New("end_tick must be >= start_tick")
	}
	m.mu.Lock()
	shape, ok := m.shapes[name]
	if !ok {
		m.mu.Unlock()
		return "", fmt.Errorf("shape %q not found", name)
	}
	shape.Moves = append(shape.Moves, Movement{
		DX:        dx,
		DY:        dy,
		DZ:        dz,
		StartTick: startTick,
		EndTick:   endTick,
	})
	m.mu.Unlock()
	return fmt.Sprintf("registered move for %s (Δ=%.3f,%.3f,%.3f ticks %d-%d)", name, dx, dy, dz, startTick, endTick), nil
}

func (m *Manager) handleSpin(args []string) (string, error) {
	if len(args) != 6 {
		return "", errors.New("usage: animate spin <name> <rad_x> <rad_y> <rad_z> <start_tick> <end_tick>")
	}
	if err := m.ensureMutable(); err != nil {
		return "", err
	}
	name := args[0]
	radX, err := strconv.ParseFloat(args[1], 64)
	if err != nil {
		return "", fmt.Errorf("invalid rad_x: %w", err)
	}
	radY, err := strconv.ParseFloat(args[2], 64)
	if err != nil {
		return "", fmt.Errorf("invalid rad_y: %w", err)
	}
	radZ, err := strconv.ParseFloat(args[3], 64)
	if err != nil {
		return "", fmt.Errorf("invalid rad_z: %w", err)
	}
	startTick, err := parseUint(args[4])
	if err != nil {
		return "", fmt.Errorf("invalid start tick: %w", err)
	}
	endTick, err := parseUint(args[5])
	if err != nil {
		return "", fmt.Errorf("invalid end tick: %w", err)
	}
	if endTick < startTick {
		return "", errors.New("end_tick must be >= start_tick")
	}
	m.mu.Lock()
	shape, ok := m.shapes[name]
	if !ok {
		m.mu.Unlock()
		return "", fmt.Errorf("shape %q not found", name)
	}
	shape.Spins = append(shape.Spins, Spin{
		RadX:      radX,
		RadY:      radY,
		RadZ:      radZ,
		StartTick: startTick,
		EndTick:   endTick,
	})
	m.mu.Unlock()
	return fmt.Sprintf("registered spin for %s (rad/tick %.3f,%.3f,%.3f ticks %d-%d)", name, radX, radY, radZ, startTick, endTick), nil
}

func (m *Manager) handleOrbit(args []string) (string, error) {
	if len(args) != 6 && len(args) != 7 {
		return "", errors.New("usage: animate orbit <name> <plane> <radius> <rad_per_tick> <start_tick> <end_tick> [phase_deg]")
	}
	if err := m.ensureMutable(); err != nil {
		return "", err
	}
	name := args[0]
	plane, err := parseOrbitPlane(args[1])
	if err != nil {
		return "", err
	}
	radius, err := strconv.ParseFloat(args[2], 64)
	if err != nil || radius <= 0 {
		return "", fmt.Errorf("invalid radius: %v", args[2])
	}
	angularSpeed, err := strconv.ParseFloat(args[3], 64)
	if err != nil {
		return "", fmt.Errorf("invalid rad_per_tick: %w", err)
	}
	startTick, err := parseUint(args[4])
	if err != nil {
		return "", fmt.Errorf("invalid start tick: %w", err)
	}
	endTick, err := parseUint(args[5])
	if err != nil {
		return "", fmt.Errorf("invalid end tick: %w", err)
	}
	if endTick < startTick {
		return "", errors.New("end_tick must be >= start_tick")
	}
	phase := 0.0
	if len(args) == 7 {
		phaseDeg, perr := strconv.ParseFloat(args[6], 64)
		if perr != nil {
			return "", fmt.Errorf("invalid phase_deg: %w", perr)
		}
		phase = phaseDeg * math.Pi / 180.0
	}
	m.mu.Lock()
	shape, ok := m.shapes[name]
	if !ok {
		m.mu.Unlock()
		return "", fmt.Errorf("shape %q not found", name)
	}
	shape.Orbits = append(shape.Orbits, Orbit{
		Plane:        plane,
		Radius:       radius,
		AngularSpeed: angularSpeed,
		Phase:        phase,
		StartTick:    startTick,
		EndTick:      endTick,
	})
	m.mu.Unlock()
	return fmt.Sprintf("registered orbit for %s (plane %s, r=%.3f, rad/tick=%.3f, ticks %d-%d)", name, plane, radius, angularSpeed, startTick, endTick), nil
}

func (m *Manager) createSphere(name string, args []string) (string, error) {
	if len(args) != 6 {
		return "", errors.New("usage: animate create <name> sphere <x> <y> <z> <radius> <filled> <block_state_id>")
	}
	center, err := parseVec3(args[0], args[1], args[2])
	if err != nil {
		return "", err
	}
	radiusFloat, err := strconv.ParseFloat(args[3], 64)
	if err != nil || radiusFloat <= 0 {
		return "", fmt.Errorf("invalid radius: %v", args[3])
	}
	radius := int(math.Round(radiusFloat))
	filled, err := parseBool(args[4])
	if err != nil {
		return "", err
	}
	blockState, err := parseBlockState(args[5])
	if err != nil {
		return "", err
	}
	cells := generateSphereOffsets(radius, filled)
	shape := &Shape{
		Name:       name,
		Type:       ShapeSphere,
		Filled:     filled,
		BlockState: blockState,
		origin:     center,
		cells:      cells,
		Radius:     float64(radius),
	}
	initial := shape.BlocksAtTick(0)
	m.storeShape(shape, initial)
	m.PlaceBlocks(initial)
	return fmt.Sprintf("created sphere %s at (%.2f,%.2f,%.2f) r=%d", name, center.X, center.Y, center.Z, radius), nil
}

func (m *Manager) createRect(name string, args []string) (string, error) {
	if len(args) != 7 {
		return "", errors.New("usage: animate create <name> rect <x> <y> <z> <width> <height> <filled> <block_state_id>")
	}
	center, err := parseVec3(args[0], args[1], args[2])
	if err != nil {
		return "", err
	}
	width, err := strconv.Atoi(args[3])
	if err != nil || width <= 0 {
		return "", fmt.Errorf("invalid width: %v", args[3])
	}
	height, err := strconv.Atoi(args[4])
	if err != nil || height <= 0 {
		return "", fmt.Errorf("invalid height: %v", args[4])
	}
	filled, err := parseBool(args[5])
	if err != nil {
		return "", err
	}
	blockState, err := parseBlockState(args[6])
	if err != nil {
		return "", err
	}
	cells := generateRectOffsets(width, height, filled)
	shape := &Shape{
		Name:       name,
		Type:       ShapeRect,
		Filled:     filled,
		BlockState: blockState,
		origin:     center,
		cells:      cells,
		Width:      width,
		Height:     height,
	}
	initial := shape.BlocksAtTick(0)
	m.storeShape(shape, initial)
	m.PlaceBlocks(initial)
	return fmt.Sprintf("created rect %s at (%.2f,%.2f,%.2f) %dx%d", name, center.X, center.Y, center.Z, width, height), nil
}

func (m *Manager) createRing(name string, args []string) (string, error) {
	if len(args) != 6 {
		return "", errors.New("usage: animate create <name> ring <x> <y> <z> <inner_radius> <outer_radius> <block_state_id>")
	}
	center, err := parseVec3(args[0], args[1], args[2])
	if err != nil {
		return "", err
	}
	innerRadius, err := strconv.ParseFloat(args[3], 64)
	if err != nil || innerRadius < 0 {
		return "", fmt.Errorf("invalid inner_radius: %v", args[3])
	}
	outerRadius, err := strconv.ParseFloat(args[4], 64)
	if err != nil || outerRadius <= 0 {
		return "", fmt.Errorf("invalid outer_radius: %v", args[4])
	}
	if innerRadius >= outerRadius {
		return "", errors.New("inner_radius must be < outer_radius")
	}
	blockState, err := parseBlockState(args[5])
	if err != nil {
		return "", err
	}
	inner := int(math.Round(innerRadius))
	outer := int(math.Round(outerRadius))
	if inner >= outer {
		inner = outer - 1
		if inner < 0 {
			inner = 0
		}
	}
	cells := generateRingOffsets(inner, outer)
	shape := &Shape{
		Name:        name,
		Type:        ShapeRing,
		BlockState:  blockState,
		origin:      center,
		cells:       cells,
		InnerRadius: float64(inner),
		Radius:      float64(outer),
	}
	initial := shape.BlocksAtTick(0)
	m.storeShape(shape, initial)
	m.PlaceBlocks(initial)
	return fmt.Sprintf("created ring %s at (%.2f,%.2f,%.2f) inner=%d outer=%d", name, center.X, center.Y, center.Z, inner, outer), nil
}

func (m *Manager) createRandom(name string, args []string) (string, error) {
	if len(args) != 11 {
		return "", errors.New("usage: animate create <name> random <start_tick> <end_tick> <min_x> <max_x> <min_y> <max_y> <min_z> <max_z> <block_state_id> <block_num> <random_seed>")
	}
	startTick, err := parseUint(args[0])
	if err != nil {
		return "", fmt.Errorf("invalid start_tick: %w", err)
	}
	endTick, err := parseUint(args[1])
	if err != nil {
		return "", fmt.Errorf("invalid end_tick: %w", err)
	}
	if endTick <= startTick {
		return "", errors.New("end_tick must be > start_tick")
	}
	minX, err := parseInt32(args[2], "min_x")
	if err != nil {
		return "", err
	}
	maxX, err := parseInt32(args[3], "max_x")
	if err != nil {
		return "", err
	}
	if minX > maxX {
		return "", errors.New("min_x must be <= max_x")
	}
	minY, err := parseInt32(args[4], "min_y")
	if err != nil {
		return "", err
	}
	maxY, err := parseInt32(args[5], "max_y")
	if err != nil {
		return "", err
	}
	if minY > maxY {
		return "", errors.New("min_y must be <= max_y")
	}
	minZ, err := parseInt32(args[6], "min_z")
	if err != nil {
		return "", err
	}
	maxZ, err := parseInt32(args[7], "max_z")
	if err != nil {
		return "", err
	}
	if minZ > maxZ {
		return "", errors.New("min_z must be <= max_z")
	}
	blockState, err := parseBlockState(args[8])
	if err != nil {
		return "", err
	}
	blockNum, err := strconv.Atoi(args[9])
	if err != nil {
		return "", fmt.Errorf("invalid block_num: %w", err)
	}
	if blockNum <= 0 {
		return "", errors.New("block_num must be > 0")
	}
	seed, err := strconv.ParseInt(args[10], 10, 64)
	if err != nil {
		return "", fmt.Errorf("invalid random_seed: %w", err)
	}

	placements, err := generateRandomPlacements(minX, maxX, minY, maxY, minZ, maxZ, blockState, blockNum, seed)
	if err != nil {
		return "", err
	}

	shape := &Shape{
		Name:       name,
		Type:       ShapeRandom,
		BlockState: blockState,
		Random: &RandomSpec{
			StartTick:  startTick,
			EndTick:    endTick,
			MinX:       minX,
			MaxX:       maxX,
			MinY:       minY,
			MaxY:       maxY,
			MinZ:       minZ,
			MaxZ:       maxZ,
			Placements: placements,
		},
	}

	initial := shape.BlocksAtTick(0)
	m.storeShape(shape, initial)
	if len(initial) > 0 {
		m.PlaceBlocks(initial)
	}

	return fmt.Sprintf(
		"created random %s ticks %d-%d in box x[%d,%d] y[%d,%d] z[%d,%d] blocks=%d seed=%d",
		name,
		startTick,
		endTick,
		minX,
		maxX,
		minY,
		maxY,
		minZ,
		maxZ,
		blockNum,
		seed,
	), nil
}

func (m *Manager) storeShape(shape *Shape, initial []hpcworld.SetblockRequest) {
	m.mu.Lock()
	shape.lastBlocks = initial
	shape.DestroyTick = 0
	shape.DestroyScheduled = false
	m.shapes[shape.Name] = shape
	m.order = insertOrdered(m.order, shape.Name)
	if len(initial) > 0 {
		if m.active == nil {
			m.active = make(map[[3]int32]int32, len(initial))
		}
		for _, b := range initial {
			key := blockKey(b)
			if b.BlockState == 0 {
				delete(m.active, key)
				continue
			}
			m.active[key] = b.BlockState
		}
	}
	m.mu.Unlock()
}

// Destory removes a shape by name. It returns true if the shape existed.
func (m *Manager) Destory(name string, tick uint) bool {
	if err := m.ensureMutable(); err != nil {
		return false
	}
	ok := m.destory(name, tick)
	return ok
}

// Destroy aliases Destory for correct spelling.
func (m *Manager) Destroy(name string, tick uint) bool {
	return m.Destory(name, tick)
}

func (m *Manager) handleDestory(args []string) (string, error) {
	if len(args) != 2 {
		return "", errors.New("usage: animate destory <name> <tick>")
	}
	if err := m.ensureMutable(); err != nil {
		return "", err
	}
	name := args[0]
	tick, err := parseUint(args[1])
	if err != nil {
		return "", fmt.Errorf("invalid tick: %w", err)
	}
	if !m.destory(name, tick) {
		return "", fmt.Errorf("shape %q not found", name)
	}
	return fmt.Sprintf("scheduled destroy for %s at tick %d", name, tick), nil
}

func (m *Manager) destory(name string, tick uint) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	shape, ok := m.shapes[name]
	if !ok {
		return false
	}
	shape.DestroyTick = tick
	shape.DestroyScheduled = true
	return true
}

func (m *Manager) handlePrecompute(args []string) (string, error) {
	if len(args) > 1 {
		return "", errors.New("usage: animate precompute [max_tick]")
	}
	var override *uint
	if len(args) == 1 {
		value, err := parseUint(args[0])
		if err != nil {
			return "", fmt.Errorf("invalid max_tick: %w", err)
		}
		override = &value
	}
	return m.precompute(override)
}

func (m *Manager) precompute(override *uint) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.precomputed {
		return "", errors.New("animations already precomputed")
	}
	if len(m.shapes) == 0 {
		return "", errors.New("no shapes to precompute")
	}
	names := m.shapeNamesLocked()
	if len(names) == 0 {
		return "", errors.New("no shapes to precompute")
	}
	maxTick := m.maxAnimationTickLocked()
	if override != nil {
		if *override < maxTick {
			return "", fmt.Errorf("max_tick %d is smaller than animation horizon %d", *override, maxTick)
		}
		maxTick = *override
	}
	baseState := m.aggregateStateLocked(names, 0)
	diffs := make([]tickDiff, maxTick+1)
	prev := baseState
	for tick := uint(1); tick <= maxTick; tick++ {
		curr := m.aggregateStateLocked(names, tick)
		diffs[tick] = diffStates(prev, curr)
		prev = curr
	}
	m.diffs = diffs
	m.active = baseState
	m.precomputed = true
	m.precomputedMaxTick = maxTick
	m.lastTick = 0
	return fmt.Sprintf("precomputed animation diffs up to tick %d", maxTick), nil
}

func (m *Manager) Tick(t uint) {
	m.mu.Lock()
	if m.precomputed {
		target := t
		if target > m.precomputedMaxTick {
			target = m.precomputedMaxTick
		}
		if target <= m.lastTick {
			m.mu.Unlock()
			return
		}
		clears := make([]hpcworld.SetblockRequest, 0)
		fills := make([]hpcworld.SetblockRequest, 0)
		if m.active == nil {
			m.active = make(map[[3]int32]int32)
		}
		for step := m.lastTick + 1; step <= target; step++ {
			diff := m.diffs[step]
			if len(diff.clears) > 0 {
				for _, b := range diff.clears {
					key := [3]int32{b.X, b.Y, b.Z}
					delete(m.active, key)
				}
				clears = append(clears, diff.clears...)
			}
			if len(diff.fills) > 0 {
				for _, b := range diff.fills {
					key := [3]int32{b.X, b.Y, b.Z}
					if b.BlockState == 0 {
						delete(m.active, key)
						continue
					}
					m.active[key] = b.BlockState
				}
				fills = append(fills, diff.fills...)
			}
		}
		m.lastTick = target
		m.mu.Unlock()
		if len(clears) > 0 {
			m.PlaceBlocks(clears)
		}
		if len(fills) > 0 {
			m.PlaceBlocks(fills)
		}
		return
	}
	if len(m.shapes) == 0 {
		prevActive := m.active
		m.active = nil
		m.mu.Unlock()
		if len(prevActive) == 0 {
			return
		}
		clears := make([]hpcworld.SetblockRequest, 0, len(prevActive))
		for key := range prevActive {
			clears = append(clears, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: 0})
		}
		if len(clears) > 0 {
			m.PlaceBlocks(clears)
		}
		return
	}
	names := m.shapeNamesLocked()
	current := make(map[[3]int32]int32, len(m.active))
	expired := make([]string, 0)
	for _, name := range names {
		shape := m.shapes[name]
		if shape == nil {
			continue
		}
		placements := shape.BlocksAtTick(t)
		shape.lastBlocks = placements
		if shape.DestroyScheduled && t >= shape.DestroyTick {
			expired = append(expired, name)
			continue
		}
		if len(placements) == 0 {
			continue
		}
		for _, b := range placements {
			if b.BlockState == 0 {
				continue
			}
			current[blockKey(b)] = b.BlockState
		}
	}
	prevActive := m.active
	m.active = current
	m.lastTick = t
	m.removeShapesLocked(expired)
	m.mu.Unlock()

	clears := make([]hpcworld.SetblockRequest, 0)
	for key := range prevActive {
		if _, ok := current[key]; ok {
			continue
		}
		clears = append(clears, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: 0})
	}
	fills := make([]hpcworld.SetblockRequest, 0)
	for key, state := range current {
		if prevState, ok := prevActive[key]; ok && prevState == state {
			continue
		}
		fills = append(fills, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: state})
	}
	if len(clears) > 0 {
		m.PlaceBlocks(clears)
	}
	if len(fills) > 0 {
		m.PlaceBlocks(fills)
	}
}

// Clear removes all shapes managed by the manager.
func (m *Manager) Clear() {
	m.mu.Lock()
	if len(m.shapes) == 0 && len(m.active) == 0 {
		m.mu.Unlock()
		return
	}
	toClear := make([]hpcworld.SetblockRequest, 0)
	seen := make(map[[3]int32]struct{})
	for _, shape := range m.shapes {
		for _, b := range shape.lastBlocks {
			key := blockKey(b)
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			toClear = append(toClear, hpcworld.SetblockRequest{X: b.X, Y: b.Y, Z: b.Z, BlockState: 0})
		}
	}
	if len(m.active) > 0 {
		for key := range m.active {
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			toClear = append(toClear, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: 0})
		}
	}
	m.shapes = make(map[string]*Shape)
	m.order = nil
	m.active = nil
	m.diffs = nil
	m.precomputed = false
	m.precomputedMaxTick = 0
	m.lastTick = 0
	m.mu.Unlock()
	if len(toClear) > 0 {
		m.PlaceBlocks(toClear)
	}
}

// MaxTick returns the largest tick where the shape's state may change.
func (s *Shape) MaxTick() uint {
	var max uint
	for _, mv := range s.Moves {
		if mv.EndTick > max {
			max = mv.EndTick
		}
	}
	for _, sp := range s.Spins {
		if sp.EndTick > max {
			max = sp.EndTick
		}
	}
	for _, ob := range s.Orbits {
		if ob.EndTick > max {
			max = ob.EndTick
		}
	}
	if s.Type == ShapeRandom && s.Random != nil {
		if s.Random.EndTick > max {
			max = s.Random.EndTick
		}
	}
	if s.DestroyScheduled && s.DestroyTick > max {
		max = s.DestroyTick
	}
	return max
}

// BlocksAtTick computes block placements for a shape at a given tick.
func (s *Shape) BlocksAtTick(t uint) []hpcworld.SetblockRequest {
	if s.DestroyScheduled && t >= s.DestroyTick {
		return nil
	}
	if s.Type == ShapeRandom && s.Random != nil {
		if t < s.Random.StartTick || t >= s.Random.EndTick {
			return nil
		}
		return s.Random.Placements
	}
	if len(s.cells) == 0 {
		return nil
	}
	pos := s.positionAtTick(t)
	rx, ry, rz := s.rotationAtTick(t)
	placements := make([]hpcworld.SetblockRequest, 0, len(s.cells))
	var sinX, cosX, sinY, cosY, sinZ, cosZ float64
	if rx != 0 {
		sinX = math.Sin(rx)
		cosX = math.Cos(rx)
	}
	if ry != 0 {
		sinY = math.Sin(ry)
		cosY = math.Cos(ry)
	}
	if rz != 0 {
		sinZ = math.Sin(rz)
		cosZ = math.Cos(rz)
	}
	for _, cell := range s.cells {
		fx := float64(cell.X)
		fy := float64(cell.Y)
		fz := float64(cell.Z)
		if rx != 0 {
			ny := fy*cosX - fz*sinX
			nz := fy*sinX + fz*cosX
			fy, fz = ny, nz
		}
		if ry != 0 {
			nx := fx*cosY + fz*sinY
			nz := -fx*sinY + fz*cosY
			fx, fz = nx, nz
		}
		if rz != 0 {
			nx := fx*cosZ - fy*sinZ
			ny := fx*sinZ + fy*cosZ
			fx, fy = nx, ny
		}
		px := pos.X + fx
		py := pos.Y + fy
		pz := pos.Z + fz
		placements = append(placements, hpcworld.SetblockRequest{
			X:          int32(math.Round(px)),
			Y:          int32(math.Round(py)),
			Z:          int32(math.Round(pz)),
			BlockState: s.BlockState,
		})
	}
	return placements
}

func (s *Shape) positionAtTick(t uint) Vec3 {
	pos := s.origin
	for _, mv := range s.Moves {
		if t < mv.StartTick {
			continue
		}
		effectiveEnd := mv.EndTick
		if t < mv.EndTick {
			effectiveEnd = t
		}
		steps := int64(effectiveEnd) - int64(mv.StartTick)
		if steps < 0 {
			steps = 0
		}
		pos.X += float64(steps) * mv.DX
		pos.Y += float64(steps) * mv.DY
		pos.Z += float64(steps) * mv.DZ
	}
	var offset Vec3
	for _, ob := range s.Orbits {
		if t < ob.StartTick {
			continue
		}
		effectiveEnd := ob.EndTick
		if t < ob.EndTick {
			effectiveEnd = t
		}
		steps := int64(effectiveEnd) - int64(ob.StartTick)
		if steps < 0 {
			continue
		}
		angle := ob.Phase + float64(steps)*ob.AngularSpeed
		switch ob.Plane {
		case OrbitPlaneXY:
			offset.X += ob.Radius * math.Cos(angle)
			offset.Y += ob.Radius * math.Sin(angle)
		case OrbitPlaneXZ:
			offset.X += ob.Radius * math.Cos(angle)
			offset.Z += ob.Radius * math.Sin(angle)
		case OrbitPlaneYZ:
			offset.Y += ob.Radius * math.Cos(angle)
			offset.Z += ob.Radius * math.Sin(angle)
		}
	}
	pos.X += offset.X
	pos.Y += offset.Y
	pos.Z += offset.Z
	return pos
}

func (s *Shape) rotationAtTick(t uint) (float64, float64, float64) {
	var rx, ry, rz float64
	for _, sp := range s.Spins {
		if t < sp.StartTick {
			continue
		}
		effectiveEnd := sp.EndTick
		if t < sp.EndTick {
			effectiveEnd = t
		}
		steps := int64(effectiveEnd) - int64(sp.StartTick)
		if steps <= 0 {
			continue
		}
		rx += float64(steps) * sp.RadX
		ry += float64(steps) * sp.RadY
		rz += float64(steps) * sp.RadZ
	}
	return rx, ry, rz
}

// PlaceBlocks sends block placements to the active block handler.
func (w *Manager) PlaceBlocks(blocks []hpcworld.SetblockRequest) {
	s := time.Now()
	hpcworld.Batch_SetBlocks(blocks)
	w.SetblockTime += time.Since(s)
}

func generateSphereOffsets(radius int, filled bool) []blockOffset {
	var cells []blockOffset
	r := float64(radius)
	for x := -radius; x <= radius; x++ {
		for y := -radius; y <= radius; y++ {
			for z := -radius; z <= radius; z++ {
				d := math.Sqrt(float64(x*x + y*y + z*z))
				if d > r+0.5 {
					continue
				}
				if !filled && d < r-0.5 {
					continue
				}
				cells = append(cells, blockOffset{X: x, Y: y, Z: z})
			}
		}
	}
	return uniqueOffsets(cells)
}

func generateRectOffsets(width, height int, filled bool) []blockOffset {
	ax := widthOffsets(width)
	az := widthOffsets(height)
	var cells []blockOffset
	for _, x := range ax {
		for _, z := range az {
			if !filled {
				if x != ax[0] && x != ax[len(ax)-1] && z != az[0] && z != az[len(az)-1] {
					continue
				}
			}
			cells = append(cells, blockOffset{X: x, Y: 0, Z: z})
		}
	}
	return uniqueOffsets(cells)
}

func generateRingOffsets(innerRadius, outerRadius int) []blockOffset {
	if outerRadius <= 0 {
		return nil
	}
	if innerRadius < 0 {
		innerRadius = 0
	}
	inner := float64(innerRadius)
	outer := float64(outerRadius)
	var cells []blockOffset
	for x := -outerRadius; x <= outerRadius; x++ {
		for z := -outerRadius; z <= outerRadius; z++ {
			d := math.Sqrt(float64(x*x + z*z))
			if d > outer+0.5 {
				continue
			}
			if d < inner-0.5 {
				continue
			}
			cells = append(cells, blockOffset{X: x, Y: 0, Z: z})
		}
	}
	return uniqueOffsets(cells)
}

func widthOffsets(size int) []int {
	half := size / 2
	start := -half
	end := half
	if size%2 == 0 {
		end = half - 1
	}
	res := make([]int, 0, size)
	for v := start; v <= end; v++ {
		res = append(res, v)
	}
	return res
}

func uniqueOffsets(offsets []blockOffset) []blockOffset {
	seen := make(map[[3]int]struct{}, len(offsets))
	result := make([]blockOffset, 0, len(offsets))
	for _, o := range offsets {
		key := [3]int{o.X, o.Y, o.Z}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, o)
	}
	return result
}

func parseOrbitPlane(token string) (OrbitPlane, error) {
	switch strings.ToLower(token) {
	case string(OrbitPlaneXY):
		return OrbitPlaneXY, nil
	case string(OrbitPlaneXZ):
		return OrbitPlaneXZ, nil
	case string(OrbitPlaneYZ):
		return OrbitPlaneYZ, nil
	default:
		return "", fmt.Errorf("invalid plane: %s", token)
	}
}

func parseVec3(xs, ys, zs string) (Vec3, error) {
	x, err := strconv.ParseFloat(xs, 64)
	if err != nil {
		return Vec3{}, fmt.Errorf("invalid x: %w", err)
	}
	y, err := strconv.ParseFloat(ys, 64)
	if err != nil {
		return Vec3{}, fmt.Errorf("invalid y: %w", err)
	}
	z, err := strconv.ParseFloat(zs, 64)
	if err != nil {
		return Vec3{}, fmt.Errorf("invalid z: %w", err)
	}
	return Vec3{X: x, Y: y, Z: z}, nil
}

func parseBool(token string) (bool, error) {
	lower := strings.ToLower(token)
	switch lower {
	case "1", "true", "t", "yes", "y":
		return true, nil
	case "0", "false", "f", "no", "n":
		return false, nil
	default:
		return false, fmt.Errorf("invalid bool: %s", token)
	}
}

func parseBlockState(token string) (int32, error) {
	value, err := strconv.ParseInt(token, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid block state id: %w", err)
	}
	return int32(value), nil
}

func parseUint(token string) (uint, error) {
	value, err := strconv.ParseUint(token, 10, 64)
	if err != nil {
		return 0, err
	}
	return uint(value), nil
}

func parseInt32(token, name string) (int32, error) {
	value, err := strconv.ParseInt(token, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", name, err)
	}
	return int32(value), nil
}

func generateRandomPlacements(minX, maxX, minY, maxY, minZ, maxZ int32, blockState int32, blockNum int, seed int64) ([]hpcworld.SetblockRequest, error) {
	spanX := uint64(int64(maxX) - int64(minX) + 1)
	spanY := uint64(int64(maxY) - int64(minY) + 1)
	spanZ := uint64(int64(maxZ) - int64(minZ) + 1)

	totalPositions := spanX * spanY * spanZ
	if uint64(blockNum) > totalPositions {
		return nil, fmt.Errorf("block_num %d exceeds available positions %d", blockNum, totalPositions)
	}

	rng := newMT19937(uint64(seed))
	placements := make([]hpcworld.SetblockRequest, 0, blockNum)
	seen := make(map[[3]int32]struct{}, blockNum)

	for len(placements) < blockNum {
		x := minX + int32(rng.nextInt(spanX))
		y := minY + int32(rng.nextInt(spanY))
		z := minZ + int32(rng.nextInt(spanZ))
		key := [3]int32{x, y, z}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		placements = append(placements, hpcworld.SetblockRequest{X: x, Y: y, Z: z, BlockState: blockState})
	}

	return placements, nil
}

// mt19937 implements the 32-bit Mersenne Twister PRNG.
type mt19937 struct {
	state [624]uint32
	index int
}

func newMT19937(seed uint64) *mt19937 {
	r := &mt19937{}
	r.seed(seed)
	return r
}

func (r *mt19937) seed(seed uint64) {
	mixed := uint32(seed ^ (seed >> 32))
	if mixed == 0 {
		mixed = 5489
	}
	r.state[0] = mixed
	for i := 1; i < len(r.state); i++ {
		prev := r.state[i-1]
		x := prev ^ (prev >> 30)
		r.state[i] = uint32((1812433253*uint64(x) + uint64(i)) & 0xffffffff)
	}
	r.index = len(r.state)
}

func (r *mt19937) twist() {
	const (
		n         = 624
		m         = 397
		matrixA   = uint32(0x9908b0df)
		upperMask = uint32(0x80000000)
		lowerMask = uint32(0x7fffffff)
	)
	for i := 0; i < n; i++ {
		x := (r.state[i] & upperMask) | (r.state[(i+1)%n] & lowerMask)
		xa := x >> 1
		if x&1 != 0 {
			xa ^= matrixA
		}
		r.state[i] = r.state[(i+m)%n] ^ xa
	}
	r.index = 0
}

func (r *mt19937) generate() uint32 {
	if r.index >= len(r.state) {
		r.twist()
	}
	y := r.state[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

func (r *mt19937) next() uint64 {
	high := uint64(r.generate())
	low := uint64(r.generate())
	return (high << 32) | low
}

func (r *mt19937) nextInt(bound uint64) uint64 {
	if bound == 0 {
		return 0
	}
	return r.next() % bound
}

func blockKey(b hpcworld.SetblockRequest) [3]int32 {
	return [3]int32{b.X, b.Y, b.Z}
}

func insertOrdered(names []string, name string) []string {
	idx := sort.SearchStrings(names, name)
	if idx < len(names) && names[idx] == name {
		return names
	}
	names = append(names, "")
	copy(names[idx+1:], names[idx:])
	names[idx] = name
	return names
}

func removeOrdered(names []string, name string) []string {
	idx := sort.SearchStrings(names, name)
	if idx >= len(names) || names[idx] != name {
		return names
	}
	copy(names[idx:], names[idx+1:])
	names = names[:len(names)-1]
	return names
}

func (m *Manager) shapeNamesLocked() []string {
	names := append([]string(nil), m.order...)
	if len(names) != len(m.shapes) {
		names = names[:0]
		for name := range m.shapes {
			names = append(names, name)
		}
		sort.Strings(names)
		m.order = append(m.order[:0], names...)
	}
	return names
}

func (m *Manager) aggregateStateLocked(names []string, t uint) map[[3]int32]int32 {
	state := make(map[[3]int32]int32)
	for _, name := range names {
		shape := m.shapes[name]
		if shape == nil {
			continue
		}
		placements := shape.BlocksAtTick(t)
		for _, b := range placements {
			if b.BlockState == 0 {
				continue
			}
			state[blockKey(b)] = b.BlockState
		}
	}
	return state
}

func diffStates(prev, curr map[[3]int32]int32) tickDiff {
	var diff tickDiff
	if len(prev) > 0 {
		diff.clears = make([]hpcworld.SetblockRequest, 0)
		for key := range prev {
			if _, ok := curr[key]; ok {
				continue
			}
			diff.clears = append(diff.clears, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: 0})
		}
	}
	if len(curr) > 0 {
		diff.fills = make([]hpcworld.SetblockRequest, 0)
		for key, state := range curr {
			if prevState, ok := prev[key]; ok && prevState == state {
				continue
			}
			diff.fills = append(diff.fills, hpcworld.SetblockRequest{X: key[0], Y: key[1], Z: key[2], BlockState: state})
		}
	}
	if len(diff.clears) == 0 {
		diff.clears = nil
	}
	if len(diff.fills) == 0 {
		diff.fills = nil
	}
	return diff
}

func (m *Manager) removeShapesLocked(names []string) {
	if len(names) == 0 {
		return
	}
	for _, name := range names {
		delete(m.shapes, name)
		m.order = removeOrdered(m.order, name)
	}
}

func (m *Manager) maxAnimationTickLocked() uint {
	var max uint
	for _, shape := range m.shapes {
		if shape == nil {
			continue
		}
		if tick := shape.MaxTick(); tick > max {
			max = tick
		}
	}
	return max
}
