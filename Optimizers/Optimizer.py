from scipy.misc import derivative
import numpy as np

from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def call(self, gradients, **kwargs):
        pass

    def partial_derivative(self, loss, parameters, data, labels):
        gradients = list()
        for index in range(parameters.shape[0]):
            gradient = self.__helper_derivative(loss, index, parameters, data, labels)
            gradients.append(gradient)

        return np.array(gradients)[:, np.newaxis]  # it converts the gradients to a column vector
        
    def __helper_derivative(self, loss, index, parameters, data, labels):
        def wraps(x):
            parameters[index] = x
            return loss(parameters, data, labels)
        return derivative(wraps, parameters[index])

    def calculate_delta(self, activation, inputs):
        outputs = list()
        for item in inputs:
            outputs.append(derivative(activation, item))

        return np.array(outputs)