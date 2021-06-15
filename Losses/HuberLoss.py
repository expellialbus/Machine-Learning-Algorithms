import numpy as np

from Losses import Loss

class HuberLoss(Loss):
    def __init__(self, delta=1):
        self.__delta = delta

    @property
    def delta(self):
        return self.__delta
    
    @delta.setter
    def delta(self, value):
        self.__delta = value

    def call(self, true_labels, predicted_labels):
        loss = list()
        for true_label, predicted_label in zip(true_labels.ravel(), predicted_labels.ravel()):
            if np.abs(predicted_label - true_label) <= self.__delta:
                loss.append((1 / 2) * np.power((predicted_label - true_label), 2))
            else:
                loss.append((self.__delta * np.abs(predicted_label - true_label)) - ((1 / 2) * np.power(self.__delta, 2)))

        loss = np.array(loss)[:, np.newaxis]    # converts the loss list to a numpy array as a column vector

        return np.mean(loss)

    def __call__(self, true_labels, predicted_labels):
        return self.call(true_labels, predicted_labels)