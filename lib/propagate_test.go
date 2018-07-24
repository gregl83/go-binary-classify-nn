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
	expected := map[string][][]float64{
		"previousActivationCostGradients": {
			{
				0.518229681743576,
				-0.19517421446563102,
			},
			{
				-0.40506361967443677,
				0.15255392842913582,
			},
			{
				2.3749682481581584,
				-0.8944539044068955,
			},
		},
		"weightCostGradients": {
			{
				-0.1007689501908629,
				1.406850960443582,
				1.6499250434203312,
			},
		},
		"biasCostGradients": {
			{
				0.506294475,
			},
		},
	}

	activationCostGradients := mat.NewDense(1, 2, []float64{
		1.62434536,
		-0.61175641,
	})

	activations := mat.NewDense(3, 2, []float64{
		-0.52817175,
		-1.07296862,
		0.86540763,
		-2.3015387,
		1.74481176,
		-0.7612069,
	})

	weights := mat.NewDense(1, 3, []float64{
		0.3190391,
		-0.24937038,
		1.46210794,
	})

	bias := mat.NewDense(1, 1, []float64{
		-2.06014071,
	})

	previousActivationCostGradients, weightCostGradients, biasCostGradients := linearBackward(
		activationCostGradients,
		activations,
		weights,
		bias,
	)

	for i := 0; i < len(expected["previousActivationCostGradients"]); i++ {
		assert.Equal(t, expected["previousActivationCostGradients"][i], previousActivationCostGradients.(*mat.Dense).RawRowView(i))
	}

	for i := 0; i < len(expected["weightCostGradients"]); i++ {
		assert.Equal(t, expected["weightCostGradients"][i], weightCostGradients.(*mat.Dense).RawRowView(i))
	}

	for i := 0; i < len(expected["biasCostGradients"]); i++ {
		assert.Equal(t, expected["biasCostGradients"][i], biasCostGradients.(*mat.Dense).RawRowView(i))
	}
}

func TestPropagateBackward(t *testing.T) {
	// todo

	assert.Equal(t, true, false)
}
