#include "LinearRegressionGD.h"

double LinearRegressionGD::MSE(double* actual, double* predicted, size_t size)
{
    double MSE = 0;
    for (size_t i = 0; i < size; i++)
        MSE += std::pow(actual[i] - predicted[i], 2);

    return MSE / size;
}

double LinearRegressionGD::CalculateM(double* actual, double* predicted, double* x, size_t size)
{
    double GD = 0;
    for (size_t i = 0; i < size; i++)
        GD += (actual[i] - predicted[i]) * x[i];

    return learning_rate * (static_cast<double>(-2) / size * GD);
}

double LinearRegressionGD::CalculateC(double* actual, double* predicted, size_t size)
{
    double GD = 0;
    for (size_t i = 0; i < size; i++)
        GD += (actual[i] - predicted[i]);

    return learning_rate * (static_cast<double>(-2) / size * GD);
}

LinearRegressionGD::LinearRegressionGD(double learning_rate)
{
    this->learning_rate = learning_rate;
}

LinearRegressionGD::~LinearRegressionGD()
{
}

double LinearRegressionGD::Fit(
    std::vector <std::vector <double>> train,
    std::vector <std::vector <double>> test,
    size_t epochs
)
{
    size_t size_train = train.size(), size_test = test.size();

    double* x_train = new double[size_train];
    double* y_train = new double[size_train];
    for (size_t i = 0; i < size_train; i++)
    {
        x_train[i] = train[i][0];
        y_train[i] = train[i][1];
    }

    double* x_test = new double[size_test];
    double* y_test = new double[size_test];
    for (size_t i = 0; i < size_test; i++)
    {
        x_test[i] = test[i][0];
        y_test[i] = test[i][1];
    }

    std::cout << "Training Started...\n";

    for (size_t i = 0; i < epochs; i++)
    {
        double* predictions = new double[size_train];

        for (size_t j = 0; j < size_train; j++)
            predictions[j] = Predict(x_train[j]);

        // Gradient Descent
        weight -= CalculateM(y_train, predictions, x_train, size_train);
        bias -= CalculateC(y_train, predictions, size_train);
    }

    double error = Eval(x_test, y_test, size_test);

    std::cout << "MSE SCORE: "  << error 
              << "\tGradient: " << weight 
              << "\tBias: "     << bias 
              << "\tLR: "       << learning_rate 
              << "\tEpochs: "   << epochs 
              << std::endl;

    std::cout << "x\ty\tprediction" << std::endl;
    for (size_t i = 0; i < size_test; i++)
    {
        std::cout << x_test[i] << "\t" << y_test[i] << "\t" << Predict(x_test[i]) << std::endl;
    }

    return error;
}

double LinearRegressionGD::Eval(double* x, double* y, size_t size)
{
    double* predictions = new double[size];
    for (size_t i = 0; i < size; i++)
        predictions[i] = Predict(x[i]);

    return MSE(y, predictions, size);
}

double LinearRegressionGD::Predict(double x)
{
    return weight * x + bias;
}
