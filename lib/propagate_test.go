package lib

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestLinearForward(t *testing.T) {
	expected := [][]float64{
		{
			3.262953358322841,
			-1.2342998768590736,
		},
	}

	activations := mat.NewDense(3, 2, []float64{
		1.62434536,
		-0.61175641,
		-0.52817175,
		-1.07296862,
		0.86540763,
		-2.3015387,
	})

	weights := mat.NewDense(1, 3, []float64{
		1.74481176,
		-0.7612069,
		0.3190391,
	})

	bias := mat.NewDense(1, 1, []float64{
		-0.24937038,
	})

	preActivated := linearForward(activations, weights, bias)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], preActivated.(*mat.Dense).RawRowView(i))
	}
}

func TestForwardPropagateSigmoid(t *testing.T) {
	expected := [][]float64{
		{
			0.9689002334527027,
			0.11013289497444277,
		},
	}

	activations := mat.NewDense(3, 2, []float64{
		-0.41675785,
		-0.05626683,
		-2.1361961,
		1.64027081,
		-1.79343559,
		-0.84174737,
	})

	weights := mat.NewDense(1, 3, []float64{
		0.50288142,
		-1.24528809,
		-1.05795222,
	})

	bias := mat.NewDense(1, 1, []float64{
		-0.90900761,
	})

	activated := PropagateForward(activations, weights, bias, "sigmoid")

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activated.(*mat.Dense).RawRowView(i))
	}
}

func TestForwardPropagateRelu(t *testing.T) {
	expected := [][]float64{
		{
			3.4389613356978117,
			0,
		},
	}

	activations := mat.NewDense(3, 2, []float64{
		-0.41675785,
		-0.05626683,
		-2.1361961,
		1.64027081,
		-1.79343559,
		-0.84174737,
	})

	weights := mat.NewDense(1, 3, []float64{
		0.50288142,
		-1.24528809,
		-1.05795222,
	})

	bias := mat.NewDense(1, 1, []float64{
		-0.90900761,
	})

	activated := PropagateForward(activations, weights, bias, "relu")

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activated.(*mat.Dense).RawRowView(i))
	}
}

func TestLinearBackward(t *testing.T) {
	assert.Equal(t, true, false)
}

func TestPropagateBackward(t *testing.T) {
	// todo

	assert.Equal(t, true, false)
}
