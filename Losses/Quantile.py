import numpy as np

from Losses import Loss

class Quantile(Loss):
    def __init__(self, gamma=0.50):
        self.__gamma = gamma

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        self.__gamma = value

    def call(self, true_labels, predicted_labels):
        greater = (predicted_labels > true_labels)
        lower = (predicted_labels <= true_labels)

        greater_sum = np.sum((self.__gamma - 1) * np.abs(predicted_labels[greater] - true_labels[greater]))
        lower_sum = np.sum(self.__gamma * np.abs(predicted_labels[lower] - true_labels[lower]))

        return greater_sum + lower_sum

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)