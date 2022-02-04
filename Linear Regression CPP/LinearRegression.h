#pragma once

#include <iostream>
#include <cmath>

class LinearRegression
{
private:
	double weight;
	double bias;

	double mean(double* x, size_t size);
	double variance(double* x, double mean, size_t size);
	double covariance(double* x, double mean_x, double* y, double mean_y, size_t size);
	double rmse(double* actual, double* predicted, size_t size);

	void GetCoefficient(double** data, size_t size);

public:
	LinearRegression();
	~LinearRegression();

	double Fit(double** train, size_t size_train, double** test, size_t size_test);
	double Predict(double x);
};

