import numpy as np

from Losses import Loss

class GeometricMeanRelativeAbsoluteError(Loss):
    def call(self, true_labels, predicted_labels):
        result = np.abs(true_labels - predicted_labels) / np.abs(true_labels - np.mean(true_labels))

        return np.power(np.prod(result), (1 / true_labels.shape[0]))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)