import numpy as np

from ..Layers import Layer
from Activations import get_activation

class Dense(Layer):
    def __init__(self, n_neurons, activation=None):
        super().__init__(n_neurons, activation)
        self._weights = None

    def initialize_weights(self, shape):
        if self._weights == None:
            self._weights = np.random.randn(shape + 1, self.n_neurons)  # the plus one for the bias node

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
            self._activation = get_activation(self._activation)

            result = self._activation(result)

        self._outputs = result

    def backward(self, delta):
        self._weights = self._outputs.T.dot()