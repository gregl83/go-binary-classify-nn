package lib

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func activate(matrix mat.Matrix, activation func(x float64) float64) mat.Dense {
	var activated mat.Dense
	activated.Apply(func(i, j int, v float64) float64 {
		return activation(v)
	}, matrix)
	return activated
}

func activatePrime(matrix, activations mat.Matrix, activation func(x, z float64) float64) mat.Dense {
	var activated mat.Dense
	activated.Apply(func(i, j int, v float64) float64 {
		return activation(v, activations.(*mat.Dense).At(i, j))
	}, matrix)
	return activated
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x, z float64) float64 {
	s := sigmoid(z)
	return x * s * (1 - s)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluPrime(x, z float64) float64 {
	if z > 0 {
		return x
	}
	return 0
}
