package lib

import (
	"gonum.org/v1/gonum/mat"
)

func columnBroadcast(bias mat.Matrix, cols int) mat.Matrix {
	b := bias.(*mat.Dense)
	rows, _ := bias.Dims()
	res := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		val := b.RawRowView(i)[0]
		for i := range row {
			row[i] = val
		}
		res.SetRow(i, row)
	}

	return res
}

func linearForward(previousActivations, weights, bias mat.Matrix) mat.Matrix {
	var preActivations mat.Dense
	preActivations.Mul(weights, previousActivations)

	_, cols := preActivations.Dims()

	var preActivationsBiased mat.Dense
	preActivationsBiased.Add(&preActivations, columnBroadcast(bias, cols))

	return &preActivationsBiased
}

// PropagateForward computes neuron activations in network layer
func PropagateForward(previousActivations, weights, bias mat.Matrix, activation string) mat.Dense {
	var activations mat.Dense

	preActivations := linearForward(previousActivations, weights, bias)

	if activation == "relu" {
		activations = activate(preActivations, relu)
	}

	if activation == "sigmoid" {
		activations = activate(preActivations, sigmoid)
	}

	return activations
}

func divide(matrix mat.Matrix, value int) mat.Dense {
	var res mat.Dense

	res.Apply(func(i, j int, v float64) float64 {
		return v / float64(value)
	}, matrix)

	return res
}

func sumRows(matrix mat.Matrix) mat.Dense {
	rows, _ := matrix.Dims()
	res := make([]float64, rows)

	for i := 0; i < rows; i++ {
		row := matrix.(*mat.Dense).RawRowView(i)
		var sum float64
		for r := 0; r < len(row); r++ {
			sum += row[r]
		}
		res[i] = sum
	}

	return *mat.NewDense(rows, 1, res)
}

func linearBackward(linearCostGradients, previousActivations, weights, bias mat.Matrix) (mat.Dense, mat.Dense, mat.Dense) {
	_, cols := previousActivations.Dims()

	var previousActivationCostGradients mat.Dense
	previousActivationCostGradients.Mul(weights.T(), linearCostGradients)

	var weightCostGradients mat.Dense
	weightCostGradients.Mul(linearCostGradients, previousActivations.T())
	weightCostGradients = divide(&weightCostGradients, cols)

	biasCostGradients := sumRows(linearCostGradients)
	biasCostGradients = divide(&biasCostGradients, cols)

	return previousActivationCostGradients, weightCostGradients, biasCostGradients
}

// PropagateBackward computes gradient of loss with respect to parameters
func PropagateBackward(activationCostGradients, activations, previousActivations, weights, bias mat.Matrix, activation string) (mat.Dense, mat.Dense, mat.Dense) {
	var linearCostGradients mat.Dense

	if activation == "relu" {
		linearCostGradients = activatePrime(activationCostGradients, activations, reluPrime)
	}

	if activation == "sigmoid" {
		linearCostGradients = activatePrime(activationCostGradients, activations, sigmoidPrime)
	}

	return linearBackward(&linearCostGradients, previousActivations, weights, bias)
}
