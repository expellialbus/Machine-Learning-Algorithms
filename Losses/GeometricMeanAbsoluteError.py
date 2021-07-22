import numpy as np

from Losses import Loss

class GeometricMeanAbsoluteError(Loss):
    def call(self, true_labels, predicted_labels):
        return np.power(np.prod(np.abs(true_labels - predicted_labels)), 1 / true_labels.shape[0])

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)