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

/*
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    return dA_prev, dW, db
 */

// PropagateBackward computes gradient of loss with respect to parameters
func PropagateBackward(activations mat.Dense, activation string) {
	var activated mat.Matrix

	if activation == "relu" {
		activated = activate(preActivated, reluPrime)
	}

	if activation == "sigmoid" {
		activated = activate(preActivated, sigmoidPrime)
	}

	preActivated := linearBackward(activations, weights, bias)

	return activated
}
