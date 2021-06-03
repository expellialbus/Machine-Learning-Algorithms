import numpy as np

from Losses import Loss

class MeanSquaredLogarithmicError(Loss):
    def call(self, true_labels, predicted_labels):
        result = np.sum(np.power(np.log((true_labels + 1) / (predicted_labels + 1)), 2))
        return result / true_labels.shape[0]

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)