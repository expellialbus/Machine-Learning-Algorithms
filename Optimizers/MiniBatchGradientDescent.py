import numpy as np

from Optimizers import Optimizer

class MiniBatchGradientDescent(Optimizer):
    def __init__(self, learning_rate, decrease_learning_rate=True, batch_size=32):
        self.__learning_rate = learning_rate
        self.__decrease_learning_rate = decrease_learning_rate
        self.__batch_size = batch_size
    
    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.__learning_rate = value;

    @property
    def decrease_learning_rate(self):
        return self.__decrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__decrease_learning_rate = value

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value


    def __learning_schedule(self, t):
        return 1 / t

    def call(self, loss, parameters, data, labels):
        random_index = np.random.randint(data.shape[0] - self.__batch_size)
        x_batch = data[random_index: random_index + self.__batch_size]
        y_batch = labels[random_index: random_index + self.__batch_size]

        gradients = self._partial_derivative(loss, parameters, x_batch, y_batch)
        new_parameters = self.__learning_rate * gradients

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)