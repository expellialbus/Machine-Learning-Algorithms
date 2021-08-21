import numpy as np

from Losses import Loss

class MedianSquarePercentageError(Loss):
    def call(self, true_labels, predicted_labels):
        error = np.abs(predicted_labels - true_labels)

        return 100 * np.median(np.square(error / np.abs(true_labels)))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)