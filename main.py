# This file is just a temporary file to try optimizers, losses, models etc.

from Optimizers.StochasticGradientDescent import StochasticGradientDescent
import numpy as np

np.random.seed(42)

X = 2 * np.random.rand(100, 3)
y = np.array(4 + (3 * X[:, 0]) + (2 * X[:, 1]) + (5 * X[:, 2]) + np.random.rand(100)).reshape(100, 1)

from Optimizers import BatchGradientDescent
from Models import LinearRegression
from Losses import MSE

lin_reg = LinearRegression(100, StochasticGradientDescent(32, 0.01), MSE())
print(lin_reg(X, y))