package lib

import (
	"gonum.org/v1/gonum/mat"
)

func Model(data, labels mat.Dense, layers []int, learningRate float64, iterations int) (Parameters, []float64) {
	costs := make([]float64, iterations)
	parameters := NewParameters(layers)

	parameters.Activations[0] = data

	for i := 0; i < iterations; iterations++ {
		PropagateForward(&parameters)

		costs[i] = Cost(&parameters.Activations[len(layers) - 1], &labels).RawRowView(0)[0]

		weightCostGradients, biasCostGradients := PropagateBackward(parameters, &labels)

		parameters.GradientUpdate(weightCostGradients, biasCostGradients, learningRate)
	}

	return parameters, costs
}