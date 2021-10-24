from abc import ABC

class Layer(ABC):
    def __init__(self, n_neurons, activation):
        self._n_neurons = n_neurons
        self._activation = activation
        self._inputs = None
        self._outputs = None

    @property
    def n_neurons(self):
        return self._n_neurons

    @n_neurons.setter
    def n_neurons(self, value):
        self._n_neurons = value

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value
