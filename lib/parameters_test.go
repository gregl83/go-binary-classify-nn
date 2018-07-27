package lib

import (
	"testing"
	"math/rand"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewParameters(t *testing.T) {
	rand.Seed(1) // testable / static results

	weightsExpected := [][][]float64{
		{},
		{
			{
				-0.01233758177597947,
				-0.0012634751070237293,
				-0.005209945711531503,
				0.022857191176995802,
				0.003228052526115799,
			},
			{
				0.005900672875996937,
				0.0015880774017643562,
				0.009892020842955818,
				-0.007312830161774791,
				0.006863807850359727,
			},
			{
				0.01585403962280623,
				0.008382059044208105,
				0.012988408475174342,
				0.0052735839305986165,
				0.007324419258045132,
			},
			{
				-0.010731798210887525,
				0.007001209024399848,
				0.004315307186960532,
				0.009996261210112625,
				-0.015239676725278933,
			},
		},
		{
			{
				-0.0031653724289408824,
				0.018894642062634817,
				0.011007291937500208,
				-0.009927431907514368,
			},
			{
				0.009897104202085316,
				-0.006152234852777649,
				-0.014350469221322283,
				-0.021514366827426447,
			},
			{
				0.0013735037357335078,
				0.004428226270265666,
				-0.008460943734555972,
				-0.000827950341361492,
			},
		},
	}

	biasExpected := [][][]float64{
		{},
		{{0}, {0}, {0}, {0}},
		{{0}, {0}, {0}},
	}

	layers := []int{5, 4, 3}
	parameters := NewParameters(layers)

	for i := 1; i < len(layers); i++ {
		nodes := layers[i]
		for n := 0; n < nodes; n++ {
			assert.Equal(t, weightsExpected[i][n], parameters.Weights[i].RawRowView(n))
			assert.Equal(t, biasExpected[i][n], parameters.Bias[i].RawRowView(n))
		}
	}
}

func TestGradientUpdate(t *testing.T) {
	rand.Seed(1) // testable / static results

	weightsExpected := [][][]float64{
		{},
		{
			{
				-0.595620697,
				-0.099917815,
				-2.145845847,
				1.82662008,
			},
			{
				-1.7656967700000001,
				-0.806271472,
				0.5111555680000001,
				-1.182588022,
			},
			{
				-1.053570403,
				-0.861285807,
				0.682840515,
				2.203745772,
			},
		},
		{
			{
				-0.555691959,
				0.035405495,
				1.3296489519999999,
			},
		},
	}

	biasExpected := [][][]float64{
		{},
		{
			{
				-0.046592414000000006,
			},
			{
				-1.288882756,
			},
			{
				0.534054956,
			},
		},
		{
			{
				-0.846107693,
			},
		},
	}

	layers := []int{4, 3, 1}
	parameters := NewParameters(layers)

	parameters.Weights = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 4, []float64{
			-0.41675785,
			-0.05626683,
			-2.1361961,
			1.64027081,
			-1.79343559,
			-0.84174737,
			0.50288142,
			-1.24528809,
			-1.05795222,
			-0.90900761,
			0.55145404,
			2.29220801,
		}),
		*mat.NewDense(1, 3, []float64{
			-0.5961597,
			-0.0191305,
			1.17500122,
		}),
	}

	parameters.Bias = []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 1, []float64{
			0.04153939,
			-1.11792545,
			0.53905832,
		}),
		*mat.NewDense(1, 1, []float64{
			-0.74787095,
		}),
	}

	weightCostGradients := []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 4, []float64{
			1.78862847,
			0.43650985,
			0.09649747,
			-1.8634927,
			-0.2773882,
			-0.35475898,
			-0.08274148,
			-0.62700068,
			-0.04381817,
			-0.47721803,
			-1.31386475,
			0.88462238,
		}),
		*mat.NewDense(1, 3, []float64{
			-0.40467741,
			-0.54535995,
			-1.54647732,
		}),
	}

	biasCostGradients := []mat.Dense{
		*mat.NewDense(0,0,nil),
		*mat.NewDense(3, 1, []float64{
			0.88131804,
			1.70957306,
			0.05003364,
		}),
		*mat.NewDense(1, 1, []float64{
			0.98236743,
		}),
	}

	learningRate := 0.1

	parameters.GradientUpdate(weightCostGradients, biasCostGradients, learningRate)

	for i := 1; i < len(layers); i++ {
		nodes := layers[i]
		for n := 0; n < nodes; n++ {
			assert.Equal(t, weightsExpected[i][n], parameters.Weights[i].RawRowView(n))
			assert.Equal(t, biasExpected[i][n], parameters.Bias[i].RawRowView(n))
		}
	}
}
