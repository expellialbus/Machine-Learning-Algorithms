import numpy as np

from Losses import Loss

class MeanSquaredError(Loss):
    def call(self, true_labels, predicted_labels):
        return np.mean(np.power((predicted_labels - true_labels), 2))
    
    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)