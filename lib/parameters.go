package lib

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func normRand(len int) []float64 {
	res := make([]float64, len)

	for i := 0; i < len; i++ {
		res[i] = rand.NormFloat64()
	}

	return res
}

// Parameters used to compute neuron activation and back propagation
type Parameters struct{
	// Layers of neural network represented by number of nodes respectively
	Layers []int
	// Weights applied to each layer feature
	Weights []mat.Dense
	// Bias applied to each neuron within a layer
	Bias []mat.Dense
	// Activations to propagate through neural layers
	Activations []mat.Dense
}

// NewParameters struct with initialized values
func NewParameters(layers []int) Parameters {
	parameters := Parameters{
		Layers: layers,
		Weights: make([]mat.Dense, len(layers)),
		Bias: make([]mat.Dense, len(layers)),
		Activations: make([]mat.Dense, len(layers)),
	}

	for i := 1; i < len(layers); i++ {
		nodes := layers[i]
		features := parameters.Layers[i - 1]
		weights := *mat.NewDense(nodes, features, normRand(features * nodes))
		weights.Scale(0.01, &weights)
		parameters.Weights[i] = weights
		parameters.Bias[i] = *mat.NewDense(nodes, 1, nil)
		parameters.Activations[i] = *mat.NewDense(nodes, 1, nil)
	}

	return parameters
}