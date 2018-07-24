package adapters

import (
	"github.com/gregl83/go-binary-classify-nn/lib"
)

// store adapter for persisting and retrieving parameters and data
type store struct {}

// CreateParameters creates and writes neural network parameters in CSV format
func (s *store) CreateParameters(parameters lib.Parameters) {
	// todo
}

// ReadParameters reads neural network parameters from CSV
func (s *store) ReadParameters() (lib.Parameters, error) {
	// todo
}

// NewStore adapter
func NewStore() *store {
	return &store{}
}