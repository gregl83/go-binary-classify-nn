package lib

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func printMatrix(name string, matrix mat.Matrix) {
	formatted := mat.Formatted(matrix, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("%s:\r\n %v\r\n", name, formatted)
}
