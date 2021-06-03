# This file is just a temporary file to try optimizers, losses, models etc.

import numpy as np

np.random.seed(42)

X_train = 2 * np.random.rand(100, 3)
y_train = np.array(4 + (3 * X_train[:, 0]) + (2 * X_train[:, 1]) + (5 * X_train[:, 2])).reshape(100, 1)

X_test = 3 * np.random.rand(10, 3)
y_test = np.array(4 + (3 * X_test[:, 0]) + (2 * X_test[:, 1]) + (5 * X_test[:, 2])).reshape(10, 1)

from Optimizers import BatchGradientDescent
from Models.LinearModels import LogisticRegression
from Losses import CosineSimilarity as CS

log_reg = LogisticRegression(2000, BatchGradientDescent(0.1), CS())
log_reg(X_train, y_train)
predictions = log_reg.inference(X_test)

print("#-------------------- Parameters ------------------------#")
print(log_reg.parameters)

print("#---------------------- Losses --------------------------#")
print(CS()(y_test, predictions))
