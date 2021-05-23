import numpy as np

from Loss import Loss

class MSE(Loss):
    def call(self, parameters, data, labels):
        result = np.sum(np.power((data.dot(parameters) - labels), 2))
        return 1 / data.shape[0] * result
    
    def __call__(self, parameters, data, labels):
        return self.call(parameters, data, labels)