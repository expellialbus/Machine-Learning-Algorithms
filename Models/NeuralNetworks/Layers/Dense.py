import numpy as np

from ..Layers import Layer

class Dense(Layer):
    def __init__(self, n_neurons, activation=None):
        super().__init__(n_neurons, activation)

        self.weights = np.random.randn(self.inputs.shape[0] + 1, n_neurons)     # the plus one for the bias node

    @property
    def weights(self):
        return self.weights

    @weights.setter
    def weights(self, value):
        self.weights = value

    def forward(self):
        X = np.c_[self.inputs, np.ones((self.inputs.shape[0], 1))]

        result = X.dot(self.weights)

        if self.activation:
            pass



