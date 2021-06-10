import numpy as np

from Optimizers import Optimizer

class AMSGrad(Optimizer):
    def __init__(self, learning_rate=0.001, beta_one=0.9, beta_two=0.999, epsilon=1e-7, decrease_learning_rate=True):
        self.__learning_rate = learning_rate
        self.__beta_one = beta_one
        self.__beta_two = beta_two
        self.__epsilon = epsilon
        self.__decrease_learning_rate = decrease_learning_rate

        self.__velocity = 0
        self.__cumulative_sum = 0

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.__learning_rate = value

    @property
    def beta_one(self):
        return self.__beta_one

    @beta_one.setter
    def beta_one(self, value):
        self.__beta_one = value

    @property
    def beta_two(self):
        return self.__beta_two

    @beta_two.setter
    def beta_two(self, value):
        self.__beta_two = value

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.__epsilon = value

    @property
    def decrease_learning_rate(self):
        return self.__decrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__decrease_learning_rate = value

    def __learning_schedule(self, t):
        return 1 / t

    def __update_velocity(self, gradients):
        self.__velocity = (self.__beta_one * self.__velocity) + ((1 - self.__beta_one) * gradients)

    def __update_cumulative_sum(self, gradients):
        cumulative_sum = (self.__beta_two * self.__cumulative_sum) + ((1 - self.__beta_two) * np.power(gradients, 2))

        if type(self.__cumulative_sum) == int:      # it will execute just for the first iteration
            self.__cumulative_sum = cumulative_sum

        else:
            self.__cumulative_sum = np.maximum(self.__cumulative_sum, cumulative_sum)

    def call(self, loss, parameters, data, labels):
        gradients = self._partial_derivative(loss, parameters, data, labels)

        self.__update_velocity(gradients)
        self.__update_cumulative_sum(gradients)

        new_parameters = (self.__learning_rate / (np.sqrt(self.__cumulative_sum) + self.__epsilon)) * self.__velocity

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)