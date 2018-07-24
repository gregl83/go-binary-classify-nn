package lib

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func log(vector mat.Vector) mat.Vector {
	var res = mat.NewVecDense(vector.Len(), nil)
	for i := 0; i < vector.Len(); i++ {
		res.SetVec(i, math.Log(vector.At(i,0)))
	}
	return res
}

func subtract(value float64, vector mat.Vector) mat.Vector {
	var res = mat.NewVecDense(vector.Len(), nil)
	for i := 0; i < vector.Len(); i++ {
		res.SetVec(i, value - vector.At(i,0))
	}
	return res
}

// Cost (cross-entropy) of predictions to labels
func Cost(predictions mat.Vector, labels mat.Vector) float64 {
	samples, _ := predictions.Dims()

	occured := mat.Dot(labels, log(predictions))
	noccured := mat.Dot(subtract(1, labels), log(subtract(1, predictions)))

	return - ((occured + noccured) / float64(samples))
}

// CostRegularized computes the cross-entropy cost then adjusts for L2 parameter regularization
/*
	python equiv cEra deep learning spec:
	cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    L2_regularization_cost = (1/m * lambd/2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy_cost + L2_regularization_cost
 */
func CostRegularized(predictions mat.Vector, labels mat.Vector) float64 {
	return Cost(predictions, labels) // FIXME + l2_reg
}