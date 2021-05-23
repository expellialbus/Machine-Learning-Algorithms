import numpy as np

np.random.seed(42)

X = 2 * np.random.rand(100, 3)
y = np.array(4 + (3 * X[:, 0]) + (2 * X[:, 1]) + (5 * X[:, 2]) + np.random.rand(100)).reshape(100, 1)


from Optimizers import StochasticGradientDescent
from Optimizers import BatchGradientDescent

def linear_regression(data, labels, n_iterations, optimizer):
    data = np.c_[np.ones((data.shape[0], 1)), data]
    parameters = np.random.rand(data.shape[1], 1)
    
    for iteration in range(n_iterations):
        parameters = parameters - optimizer(None, parameters, data, labels)

    return parameters

print(linear_regression(X, y, 1000, BatchGradientDescent(0.01)))
    