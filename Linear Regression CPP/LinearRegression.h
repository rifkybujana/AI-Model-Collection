#pragma once

#include <iostream>
#include <cmath>
#include <vector>

class LinearRegression
{
private:
	double weight;
	double bias;

	double mean(double* x, size_t size);
	double variance(double* x, double mean, size_t size);
	double covariance(double* x, double mean_x, double* y, double mean_y, size_t size);
	double rmse(double* actual, double* predicted, size_t size);

	void GetCoefficient(std::vector <std::vector <double>> data);

public:
	LinearRegression();
	~LinearRegression();

	double Fit(
		std::vector <std::vector <double>> train,
		std::vector <std::vector <double>> test
	);
	double Predict(double x);
};

