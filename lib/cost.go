package lib

import (
	"gonum.org/v1/gonum/mat"
)

// Cost (cross-entropy) of predictions to labels
func Cost(predictions, labels mat.Matrix) mat.Dense {
	samples, _ := predictions.Dims()

	predictionsScaled := log(predictions)

	var occured mat.Dense
	occured.Mul(labels.T(), &predictionsScaled)

	subtractedLabels := subtract(1, labels)
	subtractedPredictions := subtract(1, predictions)
	subtractedPredictionsScaled := log(&subtractedPredictions)

	var noccured mat.Dense
	noccured.Mul(subtractedLabels.T(), &subtractedPredictionsScaled)

	var combined mat.Dense
	combined.Add(&occured, &noccured)

	return multiply(&combined, -1 / float64(samples))
}

// CostRegularized computes the cross-entropy cost then adjusts for L2 parameter regularization
/*
	python equiv cEra deep learning spec:
	cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    L2_regularization_cost = (1/m * lambd/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy_cost + L2_regularization_cost
 */
func CostRegularized(predictions, labels mat.Matrix) mat.Dense {
	return Cost(predictions, labels) // FIXME + l2_reg
}
