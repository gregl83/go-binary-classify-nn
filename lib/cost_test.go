package lib

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestCostNotRegularized(t *testing.T) {
	expected := [][]float64{
		{0.41493159961539694},
	}

	predictions := mat.NewDense(3, 1, []float64{0.8, 0.9, 0.4})
	labels := mat.NewDense(3, 1, []float64{1, 1, 1})

	cost := Cost(predictions, labels)

	for i := 0; i < len(expected); i++ {
		assert.Equal(t, expected[i], cost.RawRowView(i))
	}
}

//func TestCostRegularized(t *testing.T) {
//	expected := [][]float64{
//		{0.41493159961539694},
//	}
//
//	predictions := mat.NewDense(3, 1, []float64{0.8, 0.9, 0.4})
//	labels := mat.NewDense(3, 1, []float64{1, 1, 1})
//
//	cost := Cost(predictions, labels)
//
//	for i := 0; i < len(expected); i++ {
//		assert.Equal(t, expected[i], cost.RawRowView(i))
//	}
//}