from Models.NeuralNetworks.Layers import Layer

class Sequential(Layer):
    def __init__(self):
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def build(self, loss, optimizer):
        pass

    def call(self, data, labels):
        self.initialize_layers(data.shape[1])

        for layer in self.layers:
            layer.inputs = data
            layer.forward()
            data = layer.outputs

    def initialize_layers(self, shape):
        for layer in self.layers:
            layer.initialize_weights(shape)
            shape = layer.n_neurons