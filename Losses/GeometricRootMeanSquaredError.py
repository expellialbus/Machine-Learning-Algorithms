import numpy as np

from Losses import Loss

class GeometricRootMeanSquaredError(Loss):
    def call(self, true_labels, predicted_labels):
        error = np.square(predicted_labels - true_labels)

        return np.power(np.prod(error), (1 / (2 * true_labels.shape[0])))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)