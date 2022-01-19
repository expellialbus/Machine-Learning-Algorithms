import numpy as np

from Optimizers import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_one=0.9, beta_two=0.999, epsilon=1e-8, decrease_learning_rate=True):
        self.__learning_rate = learning_rate
        self.__beta_one = beta_one
        self.__beta_two = beta_two
        self.__epsilon = epsilon
        self.__decrease_learning_rate = decrease_learning_rate

        self.__velocity = 0
        self.__s = 0

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

    def __update_s(self, gradients):
        self.__s = (self.__beta_two * self.__s) + ((1 - self.__s) * np.power(gradients, 2))

        return self.__s / (1 - self.__beta_two)

    def __update_velocity(self, gradients):
        self.__velocity = (self.__beta_one * self.__velocity) + ((1 - self.__beta_one) * gradients)

        return self.__velocity / (1 - self.__beta_one)

    def call(self, gradients, **kwargs):
        velocity = self.__update_velocity(gradients)
        s = self.__update_s(gradients)

        new_parameters = (self.__learning_rate / (np.sqrt(s) + self.__epsilon)) * velocity

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * gradients.shape[0])

        return new_parameters

    def __call__(self, gradients, **kwargs):
        return self.call(gradients, **kwargs)