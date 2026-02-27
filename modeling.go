// Package modeling Used for translating raw data into models
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

var (
	ErrDivisionByZero  = errors.New("division by zero")
	ErrIncorrectLength = errors.New("data slices aren't the same length")
)

type dataset struct {
	values [][]DType
	names  []string
}
type prediction struct {
	confidence DType
	results    []DType
}
type Predictor interface {
	test(dataset) (prediction, error)
	confidence() DType
}
type KNNModel struct {
	StandardizedData dataset
	Mean             []DType
	Std              []DType
	OriginalData     dataset
	Y                []DType
	K                int
}

// TODO: look into how to implement this for any kind of number, not just float64

// KNN This will run the KNN Model against data passed
func KNN(k int, x dataset, y []DType) (*KNNModel, error) {
	// standardize inputs seems smart
	var mean, std []DType
	for _, d := range x.values {
		m, s, err := standardize(d)
		if err != nil {
			log.Printf("Error Standardizing Data: %v", err)
			return nil, err
		}
		mean = append(mean, m)
		std = append(std, s)
	}

	return &KNNModel{StandardizedData: x, Mean: mean, Std: std, OriginalData: x, Y: y, K: k}, nil
}

func euclideanDistance(og []DType, newPoint []DType) (DType, error) {
	if len(og) != len(newPoint) {
		return 0.0, ErrIncorrectLength
	}
	sum := 0.0
	for i, x := range og {
		sum += (newPoint[i] - x) * (newPoint[i] - x)
	}
	d := math.Sqrt(sum)
	return d, nil
}

func standardize(data []DType) (mean DType, std DType, err error) {
	var sum, ssq DType
	n := DType(len(data))
	for _, v := range data {
		sum += v
	}
	m := sum / n
	for _, v := range data {
		ssq += math.Pow(v-m, 2)
	}
	s := math.Sqrt(ssq / n)
	if s == 0 {
		return 0, 0, ErrDivisionByZero
	}

	for i, v := range data {
		data[i] = (v - m) / std
	}
	return m, s, nil
}

func (m KNNModel) test(x dataset) (*prediction, error) {
	for i, d := range x.values {
		x.values[i] = (d - m.Mean) / m.Std
	}
	return nil, nil
}
