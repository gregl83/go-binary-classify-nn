package lib

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func printMatrix(name string, matrix mat.Matrix) {
	space := "    "
	formatted := mat.Formatted(matrix, mat.Prefix(space), mat.Squeeze())
	fmt.Printf("%s:\r\n%s%v\r\n", name, space, formatted)
}
