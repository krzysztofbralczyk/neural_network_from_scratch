import numpy as np


class Network:
    def __init__(self, input_layer_size, output_layer_size, hidden_layers):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers


inputs = [[1, 2, 3, 4, 5], [5, 2, 7, 1, 7], [4, 7, 33, 5, 7], [2, 5, 7, 4, 8]]

weights1 = [[0.2, 0.3, 0.4, 0.6, 0.7], [0.5, 0.57, 0.46, 0.51, 0.37], [0.9, 0.2, 0.5, 0.6, 0.1]]
biases1 = [1, 2, 3]

weights2 = [[0.2, 0.3, 0.4], [-0.34, 0.35, 0.17]]
biases2 = [1, 2]

first_layer_output = np.dot(inputs, np.array(weights1).T) + biases1
print(first_layer_output)
second_layer_output = np.dot(first_layer_output, np.array(weights2).T) + biases2
print(second_layer_output)
