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

func linearForward(activations, weights, bias mat.Matrix) mat.Matrix {
	var preActivated mat.Dense
	preActivated.Mul(weights, activations)

	_, cols := preActivated.Dims()

	var preActivatedBiased mat.Dense
	preActivatedBiased.Add(&preActivated, columnBroadcast(bias, cols))

	return &preActivatedBiased
}

// PropagateForward computes neuron activations in network
func PropagateForward(activations, weights, bias mat.Matrix, activation string) mat.Matrix {
	var activated mat.Matrix

	preActivated := linearForward(activations, weights, bias)

	if activation == "relu" {
		activated = activate(preActivated, relu)
	}

	if activation == "sigmoid" {
		activated = activate(preActivated, sigmoid)
	}

	return activated
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

func linearBackward(activationCostGradients, previousActivations, weights, bias mat.Matrix) (mat.Matrix, mat.Matrix, mat.Matrix) {
	_, cols := previousActivations.Dims()

	var previousActivationCostGradients mat.Dense
	previousActivationCostGradients.Mul(weights.T(), activationCostGradients)

	var weightCostGradients mat.Dense
	weightCostGradients.Mul(activationCostGradients, previousActivations.T())
	weightCostGradients = divide(&weightCostGradients, cols)

	biasCostGradients := sumRows(activationCostGradients)
	biasCostGradients = divide(&biasCostGradients, cols)

	return &previousActivationCostGradients, &weightCostGradients, &biasCostGradients
}

// PropagateBackward computes gradient of loss with respect to parameters
func PropagateBackward() {
	// todo
}
