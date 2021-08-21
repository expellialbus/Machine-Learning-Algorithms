# This file is just a temporary file to try optimizers, losses, models etc.

import numpy as np

np.random.seed(42)

X_train = 2 * np.random.rand(100, 3)
y_train = np.array(4 + (3 * X_train[:, 0]) + (2 * X_train[:, 1]) + (5 * X_train[:, 2])).reshape(100, 1)

X_test = 3 * np.random.rand(10, 3)
y_test = np.array(4 + (3 * X_test[:, 0]) + (2 * X_test[:, 1]) + (5 * X_test[:, 2])).reshape(10, 1)

"""
#visualizing of data
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

for color, dim in zip(("blue", "green", "red"), range(X_train.shape[1])):
    plt.scatter(X_train[:, dim], y_train, marker="^", color=color)
plt.show()
"""

from Optimizers import AdaMax
from Models import LinearRegression
from Losses import sMdAPE

lin_reg = LinearRegression(2000, AdaMax(), sMdAPE())
lin_reg(X_train, y_train)
predictions = lin_reg.inference(X_test)

print("#-------------------- Parameters ------------------------#")
print(lin_reg.parameters)

print("#---------------------- Losses --------------------------#")
print(sMdAPE()(y_test, predictions))
