import numpy as np

from Activations import Activation

class Relu(Activation):
    def call(self, inputs):
        return np.maximum(0, inputs)

    def __call__(self, inputs):
        return self.call(inputs)