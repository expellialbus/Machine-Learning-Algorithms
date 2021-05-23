from abc import ABC, abstractmethod, abstractproperty

class Loss(ABC):
    @abstractmethod 
    def call(self, parameters, data, labels):
        pass