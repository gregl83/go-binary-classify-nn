package lib

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func log(matrix mat.Matrix) mat.Dense {
	var res mat.Dense

	res.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, matrix)

	return res
}

func subtract(value float64, matrix mat.Matrix) mat.Dense {
	var res mat.Dense

	res.Apply(func(i, j int, v float64) float64 {
		return value - v
	}, matrix)

	return res
}

func multiply(matrix mat.Matrix, value float64) mat.Dense {
	var res mat.Dense

	res.Apply(func(i, j int, v float64) float64 {
		return v * value
	}, matrix)

	return res
}

func divide(matrix mat.Matrix, value float64) mat.Dense {
	var res mat.Dense

	res.Apply(func(i, j int, v float64) float64 {
		return v / value
	}, matrix)

	return res
}

func sumRows(matrix mat.Matrix) mat.Dense {
	rows, _ := matrix.Dims()
	res := make([]float64, rows)

	for i := 0; i < rows; i++ {
		row := matrix.(*mat.Dense).RawRowView(i)
		var sum float64
		for r := 0; r < len(row); r++ {
			sum += row[r]
		}
		res[i] = sum
	}

	return *mat.NewDense(rows, 1, res)
}

func columnBroadcast(bias mat.Matrix, cols int) mat.Dense {
	b := bias.(*mat.Dense)
	rows, _ := bias.Dims()
	res := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		val := b.RawRowView(i)[0]
		for i := range row {
			row[i] = val
		}
		res.SetRow(i, row)
	}

	return *res
}

func normRand(len int) []float64 {
	res := make([]float64, len)

	for i := 0; i < len; i++ {
		res[i] = rand.NormFloat64()
	}

	return res
}

func printMatrix(name string, matrix mat.Matrix) {
	space := "    "
	formatted := mat.Formatted(matrix, mat.Prefix(space), mat.Squeeze())
	fmt.Printf("%s:\r\n%s%v\r\n", name, space, formatted)
}

func printMatrices(name string, matrices []mat.Dense) {
	fmt.Printf("%s:\r\n", name)
	for i, matrix := range matrices {
		printMatrix(strconv.Itoa(i), &matrix)
	}
}