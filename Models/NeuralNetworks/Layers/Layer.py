from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, n_neurons, activation):
        self.n_neurons = n_neurons
        self.activation = activation

    @property
    def n_neurons(self):
        return self.n_neurons

    @n_neurons.setter
    def n_neurons(self, value):
        self.n_neurons = value

    @property
    def inputs(self):
        return self.inputs

    @inputs.setter
    def inputs(self, value):
        self.inputs = value

    @property
    def outputs(self):
        return self.outputs

    @outputs.setter
    def outputs(self, value):
        self.outputs = value

    @property
    def activation(self):
        return self.activation

    @activation.setter
    def activation(self, value):
        self.activation = value

    @abstractmethod
    def call(self, layer):
        pass
