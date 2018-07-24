package lib

import (
	"math"
	"gonum.org/v1/gonum/mat"
)

func activate(matrix mat.Matrix, activation func(x float64) float64) mat.Matrix {
	var activated mat.Dense
	activated.Apply(func(i, j int, v float64) float64 {
		return activation(v)
	}, matrix)
	return &activated
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
