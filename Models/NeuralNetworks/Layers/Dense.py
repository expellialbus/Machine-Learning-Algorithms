import numpy as np

from ..Layers import Layer
from Activations import get_activation

class Dense(Layer):
    def __init__(self, n_neurons, activation=None):
        super().__init__(n_neurons, activation)
        self._weights = None
        self._prev_weights = None

    def initialize_weights(self, shape):
        if self._weights == None:
            self._weights = np.random.randn(shape + 1, self.n_neurons)  # the plus one for the bias node

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def prev_weights(self):
        return self._prev_weights

    def forward(self):
        self.inputs = np.c_[self._inputs, np.ones((self._inputs.shape[0], 1))]

        result = self.inputs.dot(self._weights)

        # activation variable is str in the initial state
        # in the prediction phase it should be controller as not an object
        if type(self.activation) is str:
            self._activation = get_activation(self._activation)

            result = self._activation(result)

        self._outputs = result

    def backward(self, delta):
        # previous weights is used in backpropagation and weights for bias term should not be propagated
        # copying the original weights without the bias term's weights
        self._prev_weights = np.copy(np.array(self.weights[:-1, :]))
        self._weights -= self._optimizer(self._inputs.T.dot(delta), parameters=self.weights)