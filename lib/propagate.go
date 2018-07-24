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

func linearBackward() {
	// todo
}

// PropagateBackward computes gradient of loss with respect to parameters
func PropagateBackward() {
	// todo
}