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

func multiply(value float64, matrix mat.Matrix) mat.Dense {
	r, c := matrix.Dims()

	res := mat.NewDense(r, c, nil)

	res.Apply(func (i, j int, v float64) float64 {
		return v * value
	}, matrix)

	return *res
}

// Parameters used to compute neuron activation and back propagation
type Parameters struct{
	// Layers of neural network represented by number of nodes respectively
	Layers []int
	// Weights applied to each data feature
	Weights []mat.Dense
	// Bias applied to each neuron with respective weights
	Bias []mat.Dense
}

// NewParameters struct with initialized values
func NewParameters(layers []int) Parameters {
	parameters := Parameters{
		Layers: layers,
		Weights: make([]mat.Dense, len(layers)),
		Bias: make([]mat.Dense, len(layers)),
	}

	for i := 1; i < len(layers); i++ {
		nodes := layers[i]
		features := parameters.Layers[i - 1]
		weights := *mat.NewDense(nodes, features, normRand(features * nodes))
		parameters.Weights[i] = multiply(0.01, &weights)
		parameters.Bias[i] = *mat.NewDense(nodes, 1, nil)
	}

	return parameters
}