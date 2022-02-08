// Linear Regression CPP.cpp : This file contains the 'main' function. Program execution begins and ends there.

/* 
	Linear regression attempts to model the relationship between two variables by fitting a linear equation to
	observed data 
*/

#include <vector>

#include "LinearRegression.h"
#include "LinearRegressionGD.h"

int main()
{
	std::vector < std::vector <double> > train{
		{0.9, 1.4},
		{1.8, 2.6},
		{2.4, 1.0},
		{3.5, 3.7},
		{3.9, 5.5},
		{4.4, 3.2},
		{5.1, 3.0},
		{5.6, 4.9},
		{6.3, 6.3}
	};
	
	std::vector < std::vector <double> > test{
		{0.9, 1.4},
		{1.8, 2.6},
		{2.4, 1.0},
		{3.5, 3.7},
		{3.9, 5.5},
		{4.4, 3.2},
		{5.1, 3.0},
		{5.6, 4.9},
		{6.3, 6.3}
	};

	LinearRegressionGD model(0.0001);

	double rmse_score = model.Fit(train, test, 1000);
}