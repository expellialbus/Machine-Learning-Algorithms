from Optimizers import Optimizer

class NesterovAcceleratedGradient(Optimizer):
    def __init__(self, learning_rate=0.1, decrease_learning_rate=True, beta=0.9):
        self.__learning_rate = learning_rate
        self.__decrease_learning_rate = decrease_learning_rate
        self.__beta = beta
        self.__velocity = 0

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

    def __learning_scheduler(self, t):
        return 1 / t

    def __update_velocity(self, gradients):
        self.__velocity = (self.__beta * self.__velocity) + ((1 - self.__beta) * gradients)

    def call(self, loss, parameters, data, labels):
        projected_parameters = parameters - (self.__learning_rate * self.__velocity)
        gradients = self._partial_derivative(loss, projected_parameters, data, labels)

        self.__update_velocity(gradients)

        new_parameters = self.__learning_rate * self.__velocity

        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_scheduler((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)