package lib

import (
	"fmt"
	"math"

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

func printMatrix(name string, matrix mat.Matrix) {
	space := "    "
	formatted := mat.Formatted(matrix, mat.Prefix(space), mat.Squeeze())
	fmt.Printf("%s:\r\n%s%v\r\n", name, space, formatted)
}