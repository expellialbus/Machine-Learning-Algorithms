from Models.NeuralNetworks.Layers import Layer
from Losses import get_loss
from Optimizers import get_optimizer

class Sequential(Layer):
    def __init__(self):
        self.layers = list()

        self.__loss = None

    def add(self, layer):
        self.layers.append(layer)

    def build(self, loss, optimizer):
        self.__loss = get_loss(loss)
        self._optimizer = get_optimizer(optimizer)

        for layer in self.layers:
            layer.optimizer = get_optimizer(optimizer)

    def call(self, data, labels): # !!!! add epoch feature !!!!!
        self.initialize_layers(data.shape[1])
        self.inputs = data

        self.forward()
        loss = self.__loss(labels, self.outputs)
        delta = loss * self._optimizer.calculate_delta(self.layers[-1].activation, self.layers[-1].outputs)
        self.backward(delta)

    def initialize_layers(self, shape):
        for layer in self.layers:
            layer.initialize_weights(shape)
            shape = layer.n_neurons

    def forward(self):
        data = self.inputs
        for layer in self.layers:
            layer.inputs = data
            layer.forward()
            data = layer.outputs

        self.outputs = data

    def backward(self, delta):
        self.layers[-1].backward(delta)
        for layer in range(len(self.layers) - 2, (0 - 1), -1):
            delta = delta.dot(self.layers[layer + 1].prev_weights.T) * \
                    self._optimizer.calculate_delta(self.layers[layer].activation, self.layers[layer].outputs)
            self.layers[layer].backward(delta)

    def predict(self, inputs):
        self.inputs = inputs

        self.forward()

        return self.outputs