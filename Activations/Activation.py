from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def call(self, inputs):
        pass
