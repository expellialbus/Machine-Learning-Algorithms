import numpy as np

from Optimizers import Optimizer

class AdaMax(Optimizer):
    def __init__(self, learning_rate=0.002, beta_one=0.9, beta_two=0.999, decrease_learning_rate=True):
        self.__learning_rate = learning_rate
        self.__beta_one = beta_one
        self.__beta_two = beta_two
        self.__decrease_learning_rate = decrease_learning_rate

        self.__velocity = 0
        self.__norm_of_gradients = 0

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
    def beta_two(self, value):
        self.__beta_two = value

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

        return self.__velocity / (1 - self.__beta_one)

    def __update_norm_of_gradients(self, gradients):
        self.__norm_of_gradients = np.array(0).repeat(gradients.shape[0]).reshape(-1, 1)

        self.__norm_of_gradients = np.maximum((self.__beta_two * self.__norm_of_gradients), np.abs(gradients)) # vanishing gradients problem

    def call(self, gradients, **kwargs):
        velocity = self.__update_velocity(gradients)
        self.__update_norm_of_gradients(gradients)

        new_parameters = (self.__learning_rate / self.__norm_of_gradients) * velocity

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * gradients.shape[0])

        return new_parameters

    def __call__(self, gradients, **kwargs):
        return self.call(gradients, **kwargs)