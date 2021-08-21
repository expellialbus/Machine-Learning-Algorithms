import numpy as np

from Losses import Loss

class RootRelativeSquaredError(Loss):
    def call(self, true_labels, predicted_labels):
        error = np.square(predicted_labels - true_labels)
        denominator = np.square(true_labels - np.mean(true_labels))

        return np.sqrt(np.sum(error / denominator))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)