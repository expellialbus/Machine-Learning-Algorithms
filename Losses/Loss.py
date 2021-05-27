from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod 
    def call(self, true_labels, predicted_labels):
        pass