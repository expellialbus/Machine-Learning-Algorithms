import numpy as np 

from Models import Model

class LogisticRegression(Model):
    def __init__(self, training_steps, optimizer, loss):
        super().__init__(training_steps, optimizer, loss)
    
    def call(self, data, labels):
        self._initialize_model(data, labels)

        for step in range(self._training_steps):
            self._parameters -= self._optimizer(self.calculate_loss, 
                                                self._parameters,
                                                self._data,
                                                self._labels)

    def inference(self, data):
        data = self._adjust_data(data)

        return 1 / (1 + np.power(np.e, (-1 * (data.dot(self._parameters)))))
        
    def calculate_loss(self, parameters, data, labels):
        p = 1 / (1 + np.power(np.e, (-1 * (data.dot(parameters)))))
        
        return self._loss(labels, p)

    def __call__(self, data, labels):
        self.call(data, labels)