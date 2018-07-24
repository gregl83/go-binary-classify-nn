package lib

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestCostNotRegularized(t *testing.T) {
	predictions := mat.NewVecDense(3, []float64{0.8, 0.9, 0.4})
	labels := mat.NewVecDense(3, []float64{1, 1, 1})

	cost := Cost(predictions, labels)

	assert.Equal(t, 0.414931599615397, cost)
}