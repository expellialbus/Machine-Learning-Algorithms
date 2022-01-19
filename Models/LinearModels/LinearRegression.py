from Models import Model 

class LinearRegression(Model):
    def __init__(self, training_steps, optimizer, loss):
        super().__init__(training_steps, optimizer, loss)
    
    def call(self, data, labels):
        self._initialize_model(data, labels)

        for step in range(self._training_steps):
            gradients = self._optimizer.partial_derivative(self.calculate_loss,
                                                self._parameters,
                                                self._data,
                                                self._labels)

            self._parameters -= self._optimizer(gradients, parameters=self.parameters)

    def inference(self, data):
        data = self._adjust_data(data)

        return data.dot(self._parameters)

    def calculate_loss(self, parameters, data, labels):
        return self._loss(labels, data.dot(parameters))

    def __call__(self, data, labels):
        self.call(data, labels)