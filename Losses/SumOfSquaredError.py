import numpy as np

from Losses import Loss

class SumOfSquaredError(Loss):
    def call(self, true_labels, predicted_labels):
        return np.sum(np.square(predicted_labels - true_labels))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)