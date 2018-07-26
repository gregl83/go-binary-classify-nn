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

	preActivations := linearForward(activations, weights, bias)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], preActivations.(*mat.Dense).RawRowView(i))
	}
}

func TestActivateForwardSigmoid(t *testing.T) {
	expected := [][]float64{
		{
			0.9689002334527027,
			0.11013289497444277,
		},
	}

	previousActivations := mat.NewDense(3, 2, []float64{
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

	activations := activateForward(previousActivations, weights, bias, "sigmoid")

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activations.RawRowView(i))
	}
}

func TestActivateForwardRelu(t *testing.T) {
	expected := [][]float64{
		{
			3.4389613356978117,
			0,
		},
	}

	previousActivations := mat.NewDense(3, 2, []float64{
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

	activations := activateForward(previousActivations, weights, bias, "relu")

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], activations.RawRowView(i))
	}
}

func TestPropagateForward(t *testing.T) {
	assert.Equal(t, true, false)
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

	linearCostGradients := mat.NewDense(1, 2, []float64{
		1.62434536,
		-0.61175641,
	})

	preActivations := mat.NewDense(3, 2, []float64{
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
		linearCostGradients,
		preActivations,
		weights,
		bias,
	)

	for i := 0; i < len(expected["previousActivationCostGradients"]); i++ {
		assert.Equal(t, expected["previousActivationCostGradients"][i], previousActivationCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["weightCostGradients"]); i++ {
		assert.Equal(t, expected["weightCostGradients"][i], weightCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["biasCostGradients"]); i++ {
		assert.Equal(t, expected["biasCostGradients"][i], biasCostGradients.RawRowView(i))
	}
}

func TestActivateBackwardRelu(t *testing.T) {
	expected := map[string][][]float64{
		"previousActivationCostGradients": {
			{
				0.44090989260992697,
				0,
			},
			{
				0.37883605717723845,
				0,
			},
			{
				-0.229822800084214,
				0,
			},
		},
		"weightCostGradients": {
			{
				0.44513824690719245,
				0.3737141803009408,
				-0.10478988970207351,
			},
		},
		"biasCostGradients": {
			{
				-0.208378925,
			},
		},
	}

	activationCostGradients := mat.NewDense(1, 2, []float64{
		-0.41675785,
		-0.05626683,
	})

	preActivations := mat.NewDense(1, 2, []float64{
		0.04153939,
		-1.11792545,
	})

	previousActivations := mat.NewDense(3,2, []float64{
		-2.1361961,
		1.64027081,
		-1.79343559,
		-0.84174737,
		0.50288142,
		-1.24528809,
	})

	weights := mat.NewDense(1, 3, []float64{
		-1.05795222,
		-0.90900761,
		0.55145404,
	})

	bias := mat.NewDense(1, 1, []float64{
		2.29220801,
	})

	previousActivationCostGradients, weightCostGradients, biasCostGradients := activateBackward(
		activationCostGradients,
		preActivations,
		previousActivations,
		weights,
		bias,
		"relu",
	)

	for i := 0; i < len(expected["previousActivationCostGradients"]); i++ {
		assert.Equal(t, expected["previousActivationCostGradients"][i], previousActivationCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["weightCostGradients"]); i++ {
		assert.Equal(t, expected["weightCostGradients"][i], weightCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["biasCostGradients"]); i++ {
		assert.Equal(t, expected["biasCostGradients"][i], biasCostGradients.RawRowView(i))
	}
}

func TestActivateBackwardSigmoid(t *testing.T) {
	expected := map[string][][]float64{
		"previousActivationCostGradients": {
			{
				0.11017993687111528,
				0.01105339523719341,
			},
			{
				0.09466817044456259,
				0.009497234560315553,
			},
			{
				-0.057430921894111135,
				-0.005761545128443573,
			},
		},
		"weightCostGradients": {
			{
				0.10266786428377704,
				0.09778550606902146,
				-0.019680842328738218,
			},
		},
		"biasCostGradients": {
			{
				-0.05729622274827718,
			},
		},
	}

	activationCostGradients := mat.NewDense(1, 2, []float64{
		-0.41675785,
		-0.05626683,
	})

	preActivations := mat.NewDense(1, 2, []float64{
		0.04153939,
		-1.11792545,
	})

	previousActivations := mat.NewDense(3,2, []float64{
		-2.1361961,
		1.64027081,
		-1.79343559,
		-0.84174737,
		0.50288142,
		-1.24528809,
	})

	weights := mat.NewDense(1, 3, []float64{
		-1.05795222,
		-0.90900761,
		0.55145404,
	})

	bias := mat.NewDense(1, 1, []float64{
		2.29220801,
	})

	previousActivationCostGradients, weightCostGradients, biasCostGradients := activateBackward(
		activationCostGradients,
		preActivations,
		previousActivations,
		weights,
		bias,
		"sigmoid",
	)

	for i := 0; i < len(expected["previousActivationCostGradients"]); i++ {
		assert.Equal(t, expected["previousActivationCostGradients"][i], previousActivationCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["weightCostGradients"]); i++ {
		assert.Equal(t, expected["weightCostGradients"][i], weightCostGradients.RawRowView(i))
	}

	for i := 0; i < len(expected["biasCostGradients"]); i++ {
		assert.Equal(t, expected["biasCostGradients"][i], biasCostGradients.RawRowView(i))
	}
}

func TestPropagateBackward(t *testing.T) {
	expected := []map[string][][]float64{
		{
			"activationCostGradients": {
				{
					0,
					0.5225790112041578,
				},
				{
					0,
					-0.3269206014405846,
				},
				{
					0,
					-0.3207040357289928,
				},
				{
					0,
					-0.7407918690808015,
				},
			},
			"weightCostGradients": {},
			"biasCostGradients": {},
		},
		{
			"activationCostGradients": {
				{
					0.1291316177875634,
					-0.44014126700066897,
				},
				{
					-0.14175654703688387,
					0.48317296172264806,
				},
				{
					0.016637075116516804,
					-0.05670697422079575,
				},
			},
			"weightCostGradients": {
				{
					0.41010001901224874,
					0.07807203346853249,
					0.1379844368527405,
					0.10502167417988163,
				},
				{
					0,
					0,
					0,
					0,
				},
				{
					0.05283651624977053,
					0.010058654166727897,
					0.017777655698590702,
					0.013530795262454466,
				},
			},
			"biasCostGradients": {
				{
					-0.22007063350033448,
				},
				{
					0,
				},
				{
					-0.028353487110397875,
				},
			},
		},
		{
			"activationCostGradients": {
				{
					-0.5590876007916837,
					1.7746539136487123,
				},
			},
			"weightCostGradients": {
				{
					-0.39202432174003965,
					-0.1332585489800966,
					-0.04601088848081533,
				},
			},
			"biasCostGradients": {
				{
					0.1518786074265034,
				},
			},
		},
	}

	parameters := NewParameters([]int{4, 3, 1})

	parameters.Weights = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 4, []float64{
			-1.31386475,
			0.88462238,
			0.88131804,
			1.70957306,
			0.05003364,
			-0.40467741,
			-0.54535995,
			-1.54647732,
			0.98236743,
			-1.10106763,
			-1.18504653,
			-0.2056499,
		}),
		*mat.NewDense(1, 3, []float64{
			-1.02387576,
			1.12397796,
			-0.13191423,
		}),
	}

	parameters.Bias = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 1, []float64{
			1.48614836,
			0.23671627,
			-1.02378514,
		}),
		*mat.NewDense(1, 1, []float64{
			-1.62328545,
		}),
	}

	parameters.PreActivations = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 2, []float64{
			-0.7129932,
			0.62524497,
			-0.16051336,
			-0.76883635,
			-0.23003072,
			0.74505627,
		}),
		*mat.NewDense(1, 2, []float64{
			0.64667545,
			-0.35627076,
		}),
	}

	parameters.Activations = []mat.Dense{
		*mat.NewDense(4, 2, []float64{
			0.09649747,
			-1.8634927,
			-0.2773882,
			-0.35475898,
			-0.08274148,
			-0.62700068,
			-0.04381817,
			-0.47721803,
		}),
		*mat.NewDense(3, 2, []float64{
			1.97611078,
			-1.24412333,
			-0.62641691,
			-0.80376609,
			-2.41908317,
			-0.92379202,
		}),
		*mat.NewDense(1, 2, []float64{
			1.78862847,
			0.43650985,
		}),
	}

	labels := mat.NewDense(1, 2, []float64{
		1,
		0,
	})

	activationCostGradients, weightCostGradients, biasCostGradients := PropagateBackward(parameters, labels)

	for layer, gradients := range expected {
		for i := 0; i < len(gradients["activationCostGradients"]); i++ {
			assert.Equal(t, gradients["activationCostGradients"][i], activationCostGradients[layer].RawRowView(i))
		}

		for i := 0; i < len(gradients["weightCostGradients"]); i++ {
			assert.Equal(t, gradients["weightCostGradients"][i], weightCostGradients[layer].RawRowView(i))
		}

		for i := 0; i < len(gradients["biasCostGradients"]); i++ {
			assert.Equal(t, gradients["biasCostGradients"][i], biasCostGradients[layer].RawRowView(i))
		}
	}
}