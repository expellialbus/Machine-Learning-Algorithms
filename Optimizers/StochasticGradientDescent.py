import numpy as np

from Optimizer import Optimizer

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, decrease_learning_rate=True):
        super().__init__(learning_rate)
        self.__decrease_learning_rate = decrease_learning_rate 

    @property
    def decrease_learning_rate(self):
        return self.__decrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__decrease_learning_rate = value

    def call(self, loss, parameters, data, labels):
        for item in range(data.shape[0]):
            random_index = np.random.randint(data.shape[0])
            x_i = data[random_index: random_index + 1]        # it is equivalent to the data[random_idnex].reshape(1, -1)
            y_i = labels[random_index: random_index + 1]      # it is equivalent to the labels[random_index].reshape(1, -1)

            gradients = self._partial_derivative(loss, parameters, x_i, y_i)
            new_parameters = self._learning_rate * gradients

            if self.__decrease_learning_rate:
                self._learning_rate = self._learning_schedule((1 / (self._learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)