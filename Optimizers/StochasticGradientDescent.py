import numpy as np

from Optimizer import Optimizer

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, decrease_learning_rate=True):
        self.__learning_rate = learning_rate
        self.__decrease_learning_rate = decrease_learning_rate 
    
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

    def call(self, loss, parameters, data, labels):
        for item in range(data.shape[0]):
            random_index = np.random.randint(data.shape[0])
            x_i = data[random_index: random_index + 1]        # it is equivalent to the data[random_idnex].reshape(1, -1)
            y_i = labels[random_index: random_index + 1]      # it is equivalent to the labels[random_index].reshape(1, -1)

            gradients = 2 * x_i.T.dot(x_i.dot(parameters) - y_i)
            new_parameters = self.__learning_rate * gradients

            if self.__decrease_learning_rate:
                self.__learning_rate = self._learning_schedule((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)