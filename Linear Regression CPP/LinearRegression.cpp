#include "LinearRegression.h"

double LinearRegression::mean(double* x, size_t size)
{
	double sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += x[i];

	return sum / size;
}

double LinearRegression::variance(double* x, double mean, size_t size)
{
	double sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += std::pow(x[i] - mean, 2);

	return sum;
}

double LinearRegression::covariance(double* x, double mean_x, double* y, double mean_y, size_t size)
{
	double sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += (x[i] - mean_x) * (y[i] - mean_y);

	return sum;
}

double LinearRegression::rmse(double* actual, double* predicted, size_t size)
{
	double sum_error = 0;
	for (size_t i = 0; i < size; i++)
		sum_error += std::pow(predicted[i] - actual[i], 2);

	double mean_error = sum_error / size;
	return std::sqrt(mean_error);
}

void LinearRegression::GetCoefficient(double** data, size_t size)
{
	double* x = new double[size];
	double* y = new double[size];

	for (size_t i = 0; i < size; i++)
	{
		x[i] = data[i][0];
		y[i] = data[i][1];
	}

	double x_mean = mean(x, size);
	double y_mean = mean(y, size);

	weight = covariance(x, x_mean, y, y_mean, size) / variance(x, x_mean, size);
	bias = y_mean - weight * x_mean;
}

LinearRegression::LinearRegression()
{
}

LinearRegression::~LinearRegression()
{
}

double LinearRegression::Fit(double** train, size_t size_train, double** test, size_t size_test)
{
	double* predictions = new double[size_test];

	double* x = new double[size_test];
	double* y = new double[size_test];

	GetCoefficient(train, size_train);

	for (size_t i = 0; i < size_test; i++)
	{
		x[i] = test[i][0];
		y[i] = test[i][1];

		predictions[i] = Predict(x[i]);
	}

	double rmse_score = rmse(predictions, y, size_test);

	std::cout << "RMSE SCORE: " << rmse_score << "\t- Gradient: " << weight << "\t- Bias: " << bias << std::endl;
	std::cout << "x\ty\tprediction" << std::endl;
	for (size_t i = 0; i < size_test; i++)
	{
		std::cout << x[i] << "\t" << y[i] << "\t" << predictions[i] << std::endl;
	}

	return rmse_score;
}

double LinearRegression::Predict(double x)
{
	return weight * x + bias;
}
