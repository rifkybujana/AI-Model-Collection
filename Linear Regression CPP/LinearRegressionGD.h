#pragma once

#include <cmath>
#include <iostream>
#include <stdio.h>

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

	double Fit(double** train, size_t size_train, double** test, size_t size_test, size_t epochs);
	double Eval(double* x, double* y, size_t size);

	double Predict(double x);
};

