import numpy as np

from Optimizers import Optimizer

class Adadelta(Optimizer):
    def __init__(self, beta=0.95, epsilon=1e-6):
        self.__beta = beta
        self.__epsilon = epsilon
        self.__delta = 0
        self.__cumulative_sum = 0
        self.__old_parameters = 0

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

    def __update_delta(self, parameters):
        self.__delta = (self.__beta * self.__delta) + ((1 - self.__beta) * np.power(parameters, 2))

    def __update_parameters(self, parameters):
        delta_parameters = parameters - self.__old_parameters
        self.__old_parameters = parameters

        return delta_parameters

    def __update_cumulative_sum(self, gradients):
        self.__cumulative_sum = (self.__beta * self.__cumulative_sum) + ((1 - self.__beta) * np.power(gradients, 2))

    def call(self, loss, parameters, data, labels):
        gradients = self._partial_derivative(loss, parameters, data, labels)
        self.__update_cumulative_sum(gradients)

        new_parameters = (np.sqrt(self.__delta + self.__epsilon) / np.sqrt(self.__cumulative_sum + self.__epsilon)) * gradients

        delta_parameters = self.__update_parameters(parameters)
        self.__update_delta(delta_parameters)

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)

