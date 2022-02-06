// Linear Regression CPP.cpp : This file contains the 'main' function. Program execution begins and ends there.

/* 
	Linear regression attempts to model the relationship between two variables by fitting a linear equation to
	observed data 
*/

#include "LinearRegression.h"
#include "LinearRegressionGD.h"

int main()
{
	// idk how to initialize pointer to pointer array, this is the closest thing i could thinking of

	double** train = new double* [9]{
		new double[2] {0.9, 1.4},
		new double[2] {1.8, 2.6},
		new double[2] {2.4, 1.0},
		new double[2] {3.5, 3.7},
		new double[2] {3.9, 5.5},
		new double[2] {4.4, 3.2},
		new double[2] {5.1, 3.0},
		new double[2] {5.6, 4.9},
		new double[2] {6.3, 6.3}
	};

	double** test = new double* [9]{
		new double[2] {0.9, 1.4},
		new double[2] {1.8, 2.6},
		new double[2] {2.4, 1.0},
		new double[2] {3.5, 3.7},
		new double[2] {3.9, 5.5},
		new double[2] {4.4, 3.2},
		new double[2] {5.1, 3.0},
		new double[2] {5.6, 4.9},
		new double[2] {6.3, 6.3}
	};

	LinearRegressionGD model(0.0001);

	double rmse_score = model.Fit(train, 9, test, 9, 1000);

	delete[] train;
	delete[] test;
}