#pragma once

#include <vector>
#include <cmath>
#include <iostream>

class LinearRegressionGD
{
private:
	double weight = 0, bias = 0;
	double learning_rate = 0.01;

	double MSE(double* actual, double* predicted, size_t size);
	double CalculateM(double* actual, double* predicted, double* x, size_t size);
	double CalculateC(double* actual, double* predicted, size_t size);

public:
	LinearRegressionGD(double learning_rate = 0.01);
	~LinearRegressionGD();

	double Fit(
		std::vector <std::vector <double>> train, 
		std::vector <std::vector <double>> test, 
		size_t epochs
	);
	double Eval(double* x, double* y, size_t size);

	double Predict(double x);
};

