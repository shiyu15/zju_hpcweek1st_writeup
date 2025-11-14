package block

func IsAir(s StateID) bool {
	// return IsAirBlock(StateList[s])
	return s == 0
}

func IsAirBlock(b Block) bool {
	switch b.(type) {
	case Air, CaveAir, VoidAir:
		return true
	default:
		return false
	}
}
