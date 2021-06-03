import numpy as np

from Losses import Loss

class CosineSimilarity(Loss):
    def call(self, true_labels, predicted_labels):
        numerator = np.sum(predicted_labels * true_labels)
        denominator = np.square(np.sum(np.power(predicted_labels, 2))) * np.square(np.sum(np.power(true_labels, 2)))

        return numerator / denominator

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)