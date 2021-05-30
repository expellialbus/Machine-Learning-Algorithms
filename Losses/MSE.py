import numpy as np

from Losses import Loss

class MSE(Loss):
    def call(self, true_labels, predicted_labels):
        result = np.sum(np.power((predicted_labels - true_labels), 2))
        return (1 / true_labels.shape[0]) * result
    
    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)