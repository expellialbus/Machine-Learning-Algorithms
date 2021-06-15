import numpy as np

from Losses import Loss

class LogCosh(Loss):
    def call(self, true_labels, predicted_labels):
        return np.mean(np.log(np.cosh(predicted_labels - true_labels)))

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)