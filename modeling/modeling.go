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
* Let's make a map to dataset converion, probably needs to have a map input, and then a key value that specifies
 */

var ErrDependentVariableNotFound error = errors.New("invalid dependent variable key value")

type DType = float64

var (
	ErrDivisionByZero  = errors.New("division by zero")
	ErrIncorrectLength = errors.New("data slices aren't the same length")
)

type Dataset struct {
	Data          [][]DType // Holds the data values that influence Dependent
	Dependent     []DType   // Holds the dependent values
	DependentName string    // just a name to refer to dependent in print
	Names         []string  // just a name to refer to independent vars in print, comes from keys in maps if possible
}

type Prediction struct {
	Confidence DType
	Results    []DType
}
type Predictor interface {
	test(Dataset) (Prediction, error)
	confidence() DType
}
type KNNModel struct {
	StandardizedData Dataset
	Mean             []DType
	Std              []DType
	OriginalData     Dataset
	Y                []DType
	K                int
}

// TODO: look into how to implement this for any kind of number, not just float64

// KNN This will run the KNN Model against data passed
func KNN(k int, x Dataset, y []DType) (*KNNModel, error) {
	// standardize inputs seems smart
	var mean, std []DType
	for _, d := range x.Data {
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

func (m KNNModel) test(x Dataset) (*Prediction, error) {
	closest := make([]struct {
		distance DType
		value    DType
	}, m.K)
	for n, d := range x.Data {
		// here i need to initialize the avgDistance variable to be changed
		// likely i'll just keep a running sum of squared differences,
		sum, count := 0.0, 0.0
		for j, f := range d {
			// Here I need to go through each point and find the distance from the 'Mean' from the model. when the whole point has the lowest average distance, it'll be recorded
			sum += (f - m.Mean[j]) / m.Std[j]
			count += 1
		}
		sum = sum / count
		for i, d := range closest {
			if sum < d.distance {
				closest[i].distance = sum
				closest[i].value = m.Y[n]
			}
		}
	}

	// Definitely need to redesign the prediction struct, probably a list of names and a list of values? i'm not sure exactly how I'd do it.
	return nil, nil
}

func Load(dataMap map[string][]DType, dependentVariableKey string) (*Dataset, error) {
	_, ok := dataMap[dependentVariableKey]
	if !ok {
		return nil, ErrDependentVariableNotFound
	}
	// Continue here
	return nil, nil
}
