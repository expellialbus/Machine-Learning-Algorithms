import numpy as np

from Losses import Loss

class SymmetricMedianAbsolutePercentageError(Loss):
    def call(self, true_labels, predicted_labels):
        error = 2 * np.abs(predicted_labels - true_labels)
        denominator = np.abs(predicted_labels) + np.abs(true_labels)

        return 100 * np.median(error / denominator)

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)
