from Optimizers import Optimizer

class BatchGradientDescent(Optimizer):
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
        return self.__descrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__descrease_learning_rate = value

    def __learning_schedule(self, t):
        return 1 / t
    
    def call(self, loss, parameters, data, labels):
        gradients = self._partial_derivative(loss, parameters, data, labels)
        new_parameters = self.__learning_rate * gradients
        
        if self.__decrease_learning_rate:
            self.__learning_rate = self.__learning_schedule((1 / (self.__learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)

