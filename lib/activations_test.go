package lib

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestActivateRelu(t *testing.T) {
	expected := [][]float64{
		{
			3.43896131,
			0,
		},
	}

	preActivations := mat.NewDense(1, 2, []float64{
		3.43896131,
		-2.08938436,
	})

	activations := activate(preActivations, relu)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activations.(*mat.Dense).RawRowView(i))
	}
}

func TestActivateReluPrime(t *testing.T) {
	expected := [][]float64{
		{
			-0.41675785,
			0,
		},
	}

	activationGradients := mat.NewDense(1,2, []float64{
		-0.41675785,
		-0.05626683,
	})

	previousActivationGradients := mat.NewDense(1, 2, []float64{
		0.04153939,
		-1.11792545,
	})

	activationCostGradients := activatePrime(activationGradients, previousActivationGradients, reluPrime)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activationCostGradients.(*mat.Dense).RawRowView(i))
	}
}

func TestActivateSigmoid(t *testing.T) {
	expected := [][]float64{
		{
			0.9689002326783615,
			0.11013289484373436,
		},
	}

	preActivations := mat.NewDense(1, 2, []float64{
		3.43896131,
		-2.08938436,
	})

	activations := activate(preActivations, sigmoid)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activations.(*mat.Dense).RawRowView(i))
	}
}

func TestActivateSigmoidPrime(t *testing.T) {
	expected := [][]float64{
		{
			-0.1041445301481718,
			-0.010447915348382566,
		},
	}

	activationGradients := mat.NewDense(1,2, []float64{
		-0.41675785,
		-0.05626683,
	})

	previousActivationGradients := mat.NewDense(1, 2, []float64{
		0.04153939,
		-1.11792545,
	})

	activationCostGradients := activatePrime(activationGradients, previousActivationGradients, sigmoidPrime)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activationCostGradients.(*mat.Dense).RawRowView(i))
	}
}