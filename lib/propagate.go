package lib

import (
	"gonum.org/v1/gonum/mat"
)

func linearForward(previousActivations, weights, bias mat.Matrix) mat.Dense {
	var preActivations mat.Dense
	preActivations.Mul(weights, previousActivations)

	_, cols := preActivations.Dims()
	biasScaled := columnBroadcast(bias, cols)

	var preActivationsBiased mat.Dense
	preActivationsBiased.Add(&preActivations, &biasScaled)

	return preActivationsBiased
}

func activateForward(previousActivations, weights, bias mat.Matrix, activation string) (mat.Dense, mat.Dense) {
	var activations mat.Dense

	preActivations := linearForward(previousActivations, weights, bias)

	if activation == "relu" {
		activations = activate(&preActivations, relu)
	}

	if activation == "sigmoid" {
		activations = activate(&preActivations, sigmoid)
	}

	return preActivations, activations
}

// PropagateForward computes neuron activations for each network layer
func PropagateForward(parameters *Parameters) {
	layers := len(parameters.Layers)
	lastLayer := layers - 1

	for layer := 1; layer <= lastLayer; layer++ {
		previousLayer := layer - 1
		var activation string
		if layer == lastLayer {
			activation = "sigmoid"
		} else {
			activation = "relu"
		}

		parameters.PreActivations[layer], parameters.Activations[layer] = activateForward(
			&parameters.Activations[previousLayer],
			&parameters.Weights[layer],
			&parameters.Bias[layer],
			activation,
		)
	}
}

func linearBackward(preActivationCostGradients, preActivations, weights, bias mat.Matrix) (mat.Dense, mat.Dense, mat.Dense) {
	_, cols := preActivations.Dims()

	var previousActivationCostGradients mat.Dense
	previousActivationCostGradients.Mul(weights.T(), preActivationCostGradients)

	var weightCostGradients mat.Dense
	weightCostGradients.Mul(preActivationCostGradients, preActivations.T())
	weightCostGradients = divide(&weightCostGradients, float64(cols))

	biasCostGradients := sumRows(preActivationCostGradients)
	biasCostGradients = divide(&biasCostGradients, float64(cols))

	return previousActivationCostGradients, weightCostGradients, biasCostGradients
}

func activateBackward(activationCostGradients, preActivations, previousActivations, weights, bias mat.Matrix, activation string) (mat.Dense, mat.Dense, mat.Dense) {
	var preActivationCostGradients mat.Dense

	if activation == "relu" {
		preActivationCostGradients = activatePrime(activationCostGradients, preActivations, reluPrime)
	}

	if activation == "sigmoid" {
		preActivationCostGradients = activatePrime(activationCostGradients, preActivations, sigmoidPrime)
	}

	return linearBackward(&preActivationCostGradients, previousActivations, weights, bias)
}

// PropagateBackward computes gradient of loss with respect to parameters for each layer in network
func PropagateBackward(parameters Parameters, labels mat.Matrix) ([]mat.Dense, []mat.Dense, []mat.Dense) {
	layers := len(parameters.Layers)
	lastLayer := layers - 1

	activationCostGradients := make([]mat.Dense, layers)
	weightCostGradients := make([]mat.Dense, layers)
	biasCostGradients := make([]mat.Dense, layers)

	var occuredGradient mat.Dense
	occuredGradient.DivElem(labels, &parameters.Activations[lastLayer])

	noccuredLabels := subtract(1, labels)
	noccuredActivations := subtract(1, &parameters.Activations[lastLayer])

	var noccuredGradient mat.Dense
	noccuredGradient.DivElem(&noccuredLabels, &noccuredActivations)

	activationCostGradients[lastLayer].Sub(&occuredGradient, &noccuredGradient)
	activationCostGradients[lastLayer] = multiply(&activationCostGradients[lastLayer], -1)

	for layer := lastLayer; layer > 0; layer-- {
		var activation string
		if layer == lastLayer {
			activation = "sigmoid"
		} else {
			activation = "relu"
		}

		previousLayer := layer - 1 // layer or nodes to left
		activationCostGradients[previousLayer], weightCostGradients[layer], biasCostGradients[layer] = activateBackward(
			&activationCostGradients[layer],
			&parameters.PreActivations[layer],
			&parameters.Activations[previousLayer],
			&parameters.Weights[layer],
			&parameters.Bias[layer],
			activation,
		)
	}

	return activationCostGradients, weightCostGradients, biasCostGradients
}
