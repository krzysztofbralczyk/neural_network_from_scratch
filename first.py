import numpy as np


class Network:
    def __init__(self, input_layer_size, output_layer_size, hidden_layers):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers


weights = [[0.2, 0.3, 0.4, 0.6, 0.7], [0.5, 0.57, 0.46, 0.51, 0.37]]
inputs = [1, 2, 3, 4, 5]
bias = 1
output = np.dot(weights, inputs) + bias
print(output)
