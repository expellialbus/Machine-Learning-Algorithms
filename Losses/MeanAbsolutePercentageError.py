import numpy as np

from Losses import Loss

class MeanAbsolutePercentageError(Loss):
    def call(self, true_labels, predicted_labels):
        result = np.sum(np.abs((predicted_labels - true_labels) / true_labels))
        return result / true_labels.shape[0]

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)
