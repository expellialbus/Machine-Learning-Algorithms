import numpy as np

from Losses import Loss

class MedianAbsoluteError(Loss):
    def call(self, true_labels, predicted_labels):
        return np.median(np.abs(true_labels - predicted_labels))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)