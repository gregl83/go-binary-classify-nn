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

func activateForward(previousActivations, weights, bias mat.Matrix, activation string) mat.Dense {
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

// PropagateForward computes neuron activations for each network layer
func PropagateForward() {
	// todo
}

func linearBackward(linearCostGradients, previousActivations, weights, bias mat.Matrix) (mat.Dense, mat.Dense, mat.Dense) {
	_, cols := previousActivations.Dims()

	var previousActivationCostGradients mat.Dense
	previousActivationCostGradients.Mul(weights.T(), linearCostGradients)

	var weightCostGradients mat.Dense
	weightCostGradients.Mul(linearCostGradients, previousActivations.T())
	weightCostGradients = divide(&weightCostGradients, float64(cols))

	biasCostGradients := sumRows(linearCostGradients)
	biasCostGradients = divide(&biasCostGradients, float64(cols))

	return previousActivationCostGradients, weightCostGradients, biasCostGradients
}

func activateBackward(activationCostGradients, activations, previousActivations, weights, bias mat.Matrix, activation string) (mat.Dense, mat.Dense, mat.Dense) {
	var linearCostGradients mat.Dense

	if activation == "relu" {
		linearCostGradients = activatePrime(activationCostGradients, activations, reluPrime)
	}

	if activation == "sigmoid" {
		linearCostGradients = activatePrime(activationCostGradients, activations, sigmoidPrime)
	}

	return linearBackward(&linearCostGradients, previousActivations, weights, bias)
}

/*
# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    ### END CODE HERE ###

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads
 */


// PropagateBackward computes gradient of loss with respect to parameters for each layer in network
func PropagateBackward(activations, labels mat.Matrix, parameters Parameters) []mat.Dense {
	layers := len(parameters.Layers)
	//features, _ := activations.Dims()

	var occuredGradient mat.Dense
	occuredGradient.DivElem(labels, activations)

	noccuredLabels := subtract(1, labels)
	noccuredActivations := subtract(1, activations)

	var noccuredGradient mat.Dense
	noccuredGradient.DivElem(&noccuredLabels, &noccuredActivations)

	var activationGradient mat.Dense
	activationGradient.Sub(&occuredGradient, &noccuredGradient)
	activationGradient = multiply(&activationGradient, -1)

	lastLayer := layers - 1
	previousActivationCostGradients, weightCostGradients, biasCostGradients := activateBackward(
		&activationGradient,
		activations,
		&parameters.Activations[lastLayer],
		&parameters.Weights[lastLayer],
		&parameters.Bias[lastLayer],
		"sigmoid",
	)

	for layer := lastLayer - 1; layer >= 0; layer-- {
		previousActivationCostGradients, weightCostGradients, biasCostGradients := activateBackward(
			&activationGradient,
			activations,
			&parameters.Activations[lastLayer],
			&parameters.Weights[lastLayer],
			&parameters.Bias[lastLayer],
			"relu",
		)
	}

	// todo return grads
}