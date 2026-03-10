package main

import (
	"fmt"

	"github.com/ethanjameslong1/Archimedes/modeling"
)

func main() {
	trainValues := [][]float64{
		{1.1, 1.2},
		{0.9, 0.8},
		{1.0, 1.0},
		{4.9, 5.1},
		{5.0, 5.0},
		{5.2, 4.8},
	}

	dat := modeling.Dataset{
		Data:  trainValues,
		Names: []string{"feature_x", "feature_y"},
	}
	fmt.Printf("%v", dat)

	knnModel, err := modeling.KNN(2, dat)
}
