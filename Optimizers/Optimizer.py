from abc import ABC, abstractmethod, abstractproperty

class Optimizer(ABC):
    @abstractproperty
    def learning_rate(self):
        pass

    @abstractproperty
    def decrease_learning_rate(self):
        pass

    @abstractmethod
    def call(self, loss, parameter, data, labels):
        pass

    def _learning_schedule(self, t):
        return 1 / t