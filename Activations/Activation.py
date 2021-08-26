from abc import ABC, abstractmethod
from importlib import import_module

class Activation(ABC):
    @staticmethod
    def get_activation(name):
        module = import_module("Activations." + name.capitalize())
        activation = getattr(module, name.capitalize())

        return activation()

    @abstractmethod
    def call(self, inputs):
        pass
