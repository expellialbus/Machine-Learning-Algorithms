import numpy as np

from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, training_steps, optimizer, loss):
        self._training_steps = training_steps
        self._optimizer = optimizer
        self._loss = loss
        self._parameters = None

    @property 
    def training_steps(self):
        return self._training_steps
    
    @training_steps.setter
    def training_steps(self, value):
        self._training_steps = value

    @property 
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    
    @property 
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._value = value
        
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value
    
    @abstractmethod
    def call(self, data, label):
        pass

    def _initialize_model(self, data):
        new_data = np.c_[np.ones((data.shape[0], 1)), data]
        init_parameters = np.random.rand(new_data.shape[1], 1)
        
        return new_data, init_parameters

    @abstractmethod
    def calculate_loss(self, parameters, data, labels):
        pass

    def inference(self, data):
        data = np.c_[np.ones((data.shape[0], 1)), data]
        return data.dot(self._parameters)