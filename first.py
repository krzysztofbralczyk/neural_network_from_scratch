import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.15 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X = [[1, 2, 3, 4, 5],
     [5, 2, 7, 1, 7],
     [4, 7, 33, 5, 7],
     [2, 5, 7, 4, 8]]
# weights1 = [[0.2, 0.3, 0.4, 0.6, 0.7], [0.5, 0.57, 0.46, 0.51, 0.37], [0.9, 0.2, 0.5, 0.6, 0.1]]
# biases1 = [1, 2, 3]


layer_one = Layer(5, 1)
layer_one.forward(X)
print(layer_one.output)
