package lib

import (
	"gonum.org/v1/gonum/mat"
)

// Parameters used to compute neuron activation and back propagation
type Parameters struct {
	// Layers of neural network represented by number of neurons respectively
	Layers []int
	// Weights applied to each layer feature per neuron
	Weights []mat.Dense
	// Bias applied to each neuron within a layer
	Bias []mat.Dense
	// PreActivations for each layer neuron
	PreActivations []mat.Dense
	// Activations for each layer neuron
	Activations []mat.Dense
}

// GradientUpdate computes and updates weights and bias using gradient costs and weights
func (p *Parameters) GradientUpdate(weightsCostGradients, biasCostGradients []mat.Dense, learningRate float64) {
	for layer := 1; layer < len(p.Layers); layer++ {
		weightsScaled := multiply(&weightsCostGradients[layer], learningRate)
		p.Weights[layer].Sub(&p.Weights[layer], &weightsScaled)

		biasScaled := multiply(&biasCostGradients[layer], learningRate)
		p.Bias[layer].Sub(&p.Bias[layer], &biasScaled)
	}
}

// NewParameters struct with initialized values
func NewParameters(layers []int) Parameters {
	parameters := Parameters{
		Layers:         layers,
		Weights:        make([]mat.Dense, len(layers)),
		Bias:           make([]mat.Dense, len(layers)),
		PreActivations: make([]mat.Dense, len(layers)),
		Activations:    make([]mat.Dense, len(layers)),
	}

	for i := 1; i < len(layers); i++ {
		nodes := layers[i]
		features := parameters.Layers[i-1]
		weights := *mat.NewDense(nodes, features, normRand(features*nodes))
		weights.Scale(0.01, &weights)
		parameters.Weights[i] = weights
		parameters.Bias[i] = *mat.NewDense(nodes, 1, nil)
		parameters.PreActivations[i] = *mat.NewDense(nodes, 1, nil)
		parameters.Activations[i] = *mat.NewDense(nodes, 1, nil)
	}

	return parameters
}
