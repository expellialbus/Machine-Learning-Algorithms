from Optimizer import Optimizer

class BatchGradientDescent(Optimizer):
    def __init__(self, learning_rate, decrease_learning_rate=True):
        super().__init__(learning_rate)
        self.__decrease_learning_rate = decrease_learning_rate

    @property
    def decrease_learning_rate(self):
        return self.__descrease_learning_rate

    @decrease_learning_rate.setter
    def decrease_learning_rate(self, value):
        self.__descrease_learning_rate = value

    def call(self, loss, parameters, data, labels):
        gradients = self._partial_derivative(loss, parameters, data, labels)
        new_parameters = self._learning_rate * gradients
        
        if self.__decrease_learning_rate:
            self._learning_rate = self._learning_schedule((1 / (self._learning_rate + 1)) * data.shape[0])

        return new_parameters

    def __call__(self, loss, parameters, data, labels):
        return self.call(loss, parameters, data, labels)

