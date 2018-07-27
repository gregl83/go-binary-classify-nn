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
		assert.Equal(t, expected[i], preActivations.RawRowView(i))
	}
}

func TestActivateForwardSigmoid(t *testing.T) {
	expected := map[string][][]float64{
		"preActivations": {
			{
				3.4389613356978117,
				-2.0893843586662904,
			},
		},
		"activations": {
			{
				0.9689002334527027,
				0.11013289497444277,
			},
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

	preActivations, activations := activateForward(previousActivations, weights, bias, "sigmoid")

	for i := 0; i < len(expected["preActivations"]); i++ {
		assert.Equal(t, expected["preActivations"][i], preActivations.RawRowView(i))
	}

	for i := 0; i < len(expected["activations"]); i++ {
		assert.Equal(t, expected["activations"][i], activations.RawRowView(i))
	}
}

func TestActivateForwardRelu(t *testing.T) {
	expected := map[string][][]float64{
		"preActivations": {
			{
				3.4389613356978117,
				-2.0893843586662904,
			},
		},
		"activations": {
			{
				3.4389613356978117,
				0,
			},
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

	preActivations, activations := activateForward(previousActivations, weights, bias, "relu")

	for i := 0; i < len(expected["preActivations"]); i++ {
		assert.Equal(t, expected["preActivations"][i], preActivations.RawRowView(i))
	}

	for i := 0; i < len(expected["activations"]); i++ {
		assert.Equal(t, expected["activations"][i], activations.RawRowView(i))
	}
}

func TestPropagateForward(t *testing.T) {
	expected := []map[string][][]float64{
		{
			"preActivations": {},
			"activations": {
				{
					-0.31178367,
					0.72900392,
					0.21782079,
					-0.8990918,
				},
				{
					-2.48678065,
					0.91325152,
					1.12706373,
					-1.51409323,
				},
				{
					1.63929108,
					-0.4298936,
					2.63128056,
					0.60182225,
				},
				{
					-0.33588161,
					1.23773784,
					0.11112817,
					0.12915125,
				},
				{
					0.07612761,
					-0.15512816,
					0.63422534,
					0.810655,
				},
			},
		},
		{
			"preActivations": {
				{
					-5.2382571105871705,
					3.180401352320523,
					0.40745010450572083,
					-1.886127206781812,
				},
				{
					-2.773582360856901,
					-0.5617731556620513,
					3.1814162220477815,
					-0.9920943234869852,
				},
				{
					4.185009161731126,
					-1.7800690857353927,
					-0.14502619481868395,
					2.7214163884145224,
				},
				{
					5.058508017587575,
					-1.2567408221128744,
					-3.545666565291074,
					3.8232185205319755,
				},

			},
			"activations": {
				{
					0,
					3.180401352320523,
					0.40745010450572083,
					0,
				},
				{
					0,
					0,
					3.1814162220477815,
					0,
				},
				{
					4.185009161731126,
					0,
					0,
					2.7214163884145224,
				},
				{
					5.058508017587575,
					0,
					0,
					3.8232185205319755,
				},
			},
		},
		{
			"preActivations": {
				{
					2.2644603043092077,
					1.0997129756597923,
					-2.902980248526238,
					1.5403633530851415,
				},
				{
					6.337225701287243,
					-2.3811624660405943,
					-4.11228806440212,
					4.4858238344688,
				},
				{
					10.375083399778687,
					-0.6659146592907392,
					1.636351839685268,
					8.178701681762597,
				},
			},
			"activations": {
				{
					2.2644603043092077,
					1.0997129756597923,
					0,
					1.5403633530851415,
				},
				{
					6.337225701287243,
					0,
					0,
					4.4858238344688,
				},
				{
					10.375083399778687,
					0,
					1.636351839685268,
					8.178701681762597,
				},
			},
		},
		{
			"preActivations": {
				{
					-3.1986467141819155,
					0.8711705474068692,
					-1.402978630810819,
					-3.0031943133127355,
				},
			},
			"activations": {
				{
					0.039216681107889215,
					0.7049892061769997,
					0.19734387312466642,
					0.047281773214557316,
				},
			},
		},
	}

	parameters := NewParameters([]int{4, 3, 1, 1})

	parameters.Weights = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(4, 5, []float64{
			0.35480861,
			1.81259031,
			-1.3564758,
			-0.46363197,
			0.82465384,
			-1.17643148,
			1.56448966,
			0.71270509,
			-0.1810066,
			0.53419953,
			-0.58661296,
			-1.48185327,
			0.85724762,
			0.94309899,
			0.11444143,
			-0.02195668,
			-2.12714455,
			-0.83440747,
			-0.46550831,
			0.23371059,
		}),
		*mat.NewDense(3, 4, []float64{
			-0.12673638,
			-1.36861282,
			1.21848065,
			-0.85750144,
			-0.56147088,
			-1.0335199,
			0.35877096,
			1.07368134,
			-0.37550472,
			0.39636757,
			-0.47144628,
			2.33660781,
		}),
		*mat.NewDense(1, 3, []float64{
			0.9398248,
			0.42628539,
			-0.75815703,
		}),
	}

	parameters.Bias = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(4, 1, []float64{
			1.38503523,
			-0.51962709,
			-0.78015214,
			0.95560959,
		}),
		*mat.NewDense(3, 1, []float64{
			1.50278553,
			-0.59545972,
			0.52834106,
		}),
		*mat.NewDense(1, 1, []float64{
			-0.16236698,
		}),
	}

	parameters.Activations = []mat.Dense{
		*mat.NewDense(5, 4, []float64{
			-0.31178367,
			0.72900392,
			0.21782079,
			-0.8990918,
			-2.48678065,
			0.91325152,
			1.12706373,
			-1.51409323,
			1.63929108,
			-0.4298936,
			2.63128056,
			0.60182225,
			-0.33588161,
			1.23773784,
			0.11112817,
			0.12915125,
			0.07612761,
			-0.15512816,
			0.63422534,
			0.810655,
		}),
		*mat.NewDense(0, 0, nil),
		*mat.NewDense(0, 0, nil),
		*mat.NewDense(0, 0, nil),
	}

	PropagateForward(&parameters)

	for layer, activations := range expected {
		for i := 0; i < len(activations["preActivations"]); i++ {
			assert.Equal(t, activations["preActivations"][i], parameters.PreActivations[layer].RawRowView(i))
		}

		for i := 0; i < len(activations["activations"]); i++ {
			assert.Equal(t, activations["activations"][i], parameters.Activations[layer].RawRowView(i))
		}
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
			"weightCostGradients": {},
			"biasCostGradients": {},
		},
		{
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

	weightCostGradients, biasCostGradients := PropagateBackward(parameters, labels)

	for layer, gradients := range expected {
		for i := 0; i < len(gradients["weightCostGradients"]); i++ {
			assert.Equal(t, gradients["weightCostGradients"][i], weightCostGradients[layer].RawRowView(i))
		}

		for i := 0; i < len(gradients["biasCostGradients"]); i++ {
			assert.Equal(t, gradients["biasCostGradients"][i], biasCostGradients[layer].RawRowView(i))
		}
	}
}