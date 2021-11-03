# This file is just a temporary file to try optimizers, losses, models etc. for code errors

import numpy as np

np.random.seed(42)

X_train = 2 * np.random.rand(100, 3)
y_train = np.array(4 + (3 * X_train[:, 0]) + (2 * X_train[:, 1]) + (5 * X_train[:, 2])).reshape(100, 1)

X_test = 3 * np.random.rand(10, 3)
y_test = np.array(4 + (3 * X_test[:, 0]) + (2 * X_test[:, 1]) + (5 * X_test[:, 2])).reshape(10, 1)


#visualizing of data
# import matplotlib as mpl
# mpl.use("Qt5Agg")
# import matplotlib.pyplot as plt
#
# for color, dim in zip(("blue", "green", "red"), range(X_train.shape[1])):
#     plt.scatter(X_train[:, dim], y_train, marker="^", color=color)
# plt.show()


from Models.NeuralNetworks import Layers
from Models.NeuralNetworks import Sequential

model = Sequential()
model.add(Layers.Dense(5, "Relu"))
model.add(Layers.Dense(10, "Relu"))
model.build("MeanSquaredError", "Adam")
model.call(X_train, y_train)
print(model.layers[1].inputs.shape)
print(model.layers[0].inputs.shape)
print(model.layers[0].outputs.shape)


# from Models.LinearModels import LinearRegression
# from Optimizers import Adam
# from Losses import MAE
#
# model = LinearRegression(2000, Adam(), MAE())
# model(X_train, y_train)
# predictions = model.inference(X_test)
# print(MAE()(y_test, predictions))
#
# predictions = model.inference(X_test)
# for index in range(len(y_test)):
#     print(y_test[index], predictions[index])