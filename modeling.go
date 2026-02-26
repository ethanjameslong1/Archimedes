// Package modeling: Used for translating raw data into models
package modeling

import (
	"errors"
	"log"
	"math"
)

/*
* NOTE:
* I think that names should be an optional category (see statmodel package for reference) and when it's not present or too short it'll be populated with x0 - xi
*
* NOTE:
* I think an interface for each model seems to be smart, having it implement the various related functions for data. Then it can just be a stru
 */

type DType = float64

var ErrDivisionByZero = errors.New("division by zero")

type dataset struct {
	values [][]DType
	names  []string
}
type prediction struct {
	pvalue []DType
	result []DType
}
type Predictor interface {
	test(dataset) prediction
	confidence() DType
}
type KNNModel struct {
	StandardizedData dataset
	Y                []DType
	K                int
}

// TODO: look into how to implement this for any kind of number, not just float64

// KNN This will run the KNN Model against data passed
func knn(k int, x dataset, y []DType) (*KNNModel, error) {
	// K and a test observation x0 , the KNN classifier first identifies the
	// K points in the training data that are closest to x0 , represented by N0 .
	// It then estimates the conditional probability for class j as the fraction of
	// points in N0 whose response values equal j: Pr(Y = j|X = x0 ) = 1 0 I(yi = j).
	// x [[4,5,6,4,5,1,1], [4,4,4,9,9,9,8]]
	// y [16, 20, 28, 36, 45, 9, 8]

	// standardize inputs seems smart
	for i, d := range x.values {
		err := standardize(d)
		if err != nil {
			log.Printf("Bad Data at index %d, or something else: %v", i, err)
			return nil, err
		}
	}

	return &KNNModel{StandardizedData: x, Y: y, K: k}, nil
}

func standardize(data []DType) error {
	var sum, ssq DType
	n := DType(len(data))
	for _, v := range data {
		sum += v
	}
	mean := sum / n
	for _, v := range data {
		ssq += math.Pow(v-mean, 2)
	}
	std := math.Sqrt(ssq / n)
	if std == 0 {
		return ErrDivisionByZero
	}

	for i, v := range data {
		data[i] = (v - mean) / std
	}
	return nil
}
