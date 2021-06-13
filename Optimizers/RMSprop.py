import numpy as np

from Optimizers import Optimizer

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decrease_learning_rate=True, beta=0.9, epsilon=1e-6):
        self.__learning_rate = learning_rate
        self.__decrease_learning_rate = decrease_learning_rate
        self.__beta = beta
        self.__epsilon = epsilon
        self.__s = 0

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.__learning_rate = value

    @property
    def decrease_learning_rate(self):
        return self.__decrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__decrease_learning_rate = value

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, value):
        self.__beta = value

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.__epsilon = value

    def __learning_schedule(self, t):
        return 1 / t

    def __update_s(self, gradients):
        self.__s = (self.__beta * self.__s) + ((1 - self.__beta) * np.power(gradients, 2))

    def call(self, loss, parameters, data, labels):
        gradients = self._partial_derivative(loss, parameters, data, labels)
        self.__update_s(gradients)

        new_parameters = (self.__learning_rate / np.sqrt(self.__s + self.__epsilon)) * gradients

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)

