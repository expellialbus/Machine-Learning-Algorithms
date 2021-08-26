import numpy as np

from ..Layers import Layer
from Activations import Activation

class Dense(Layer):
    def __init__(self, n_neurons, activation=None):
        super().__init__(n_neurons, activation)

        self._weights = np.random.randn(self._inputs.shape[1] + 1, n_neurons)     # the plus one for the bias node

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def forward(self):
        X = np.c_[self._inputs, np.ones((self._inputs.shape[0], 1))]

        result = X.dot(self._weights)

        if self.activation:
            activation = Activation.get_activation(self._activation)

            result = activation(result)

        self._outputs = result


