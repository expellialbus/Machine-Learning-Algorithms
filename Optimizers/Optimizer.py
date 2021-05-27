from scipy.misc import derivative
import numpy as np

from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @abstractmethod
    def call(self, loss, parameter, data, labels):
        pass

    def _partial_derivative(self, loss, parameters, data, labels):
        gradients = list()
        for index in range(parameters.shape[0]):
            gradient = self.__helper_derivative(loss, index, parameters, data, labels)
            gradients.append(gradient)

        return np.array(gradients)[:, np.newaxis]  # converts gradients from (4,) shape to (4, 1) shape
        
    def __helper_derivative(self, loss, index, parameters, data, labels):
        def wraps(x):
            parameters[index] = x
            return loss(parameters, data, labels)
        return derivative(wraps, parameters[index])

    def _learning_schedule(self, t):
        return 1 / t
