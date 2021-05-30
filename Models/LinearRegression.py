from Models import Model 

class LinearRegression(Model):
    def __init__(self, training_steps, optimizer, loss):
        super().__init__(training_steps, optimizer, loss)
    
    def call(self, data, labels):
        data, parameters = self._initialize_model(data)

        for step in range(self._training_steps):
            parameters -= self._optimizer(self.calculate_loss, parameters, data, labels)

        self._parameters = parameters

    def calculate_loss(self, parameters, data, labels):
        return self._loss(labels, data.dot(parameters))

    def __call__(self, data, labels):
        self.call(data, labels)