import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from PIL import Image


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.15 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def test(self):
        print("layer object is working")
        print(self.weights)
        print(self.biases)
        print()


class Network:
    def __init__(self, layers_size, X, y, iterations, learning_rate):
        self.X = X
        self.y = y
        self.layers_size = layers_size
        self.layers = []

    def train(self):
        for i in self.iterations:
            self.feedforward()
            self.calculate_cost()
            self.calculate_gradient()
            self.backpropagation()

    def create_structure(self):
        for idx, num in enumerate(self.layers_size[1:]):
            # print(num, end=' ')
            # print(idx)
            self.layers.append(Layer(self.layers_size[idx], num))

    def test_structure(self):
        for layer in self.layers:
            layer.test()

    def activation_function(self, number):
        return 1/(1+np.exp(number))


# images = []
# results = []
# for i in range(10):
#     images.append(cv2.resize(cv2.imread("./assets/cat.jpg"), (250, 250)))
#     images.append(cv2.resize(cv2.imread("./assets/dog.jpg"), (250, 250)))
# images = np.array(images)
# # print(images)
# # print(images.shape)
# images_flattened = images.reshape(images.shape[0], -1).T
# images_flattened = images_flattened / 255
# print(images_flattened.shape)

network = Network([9, 5, 3, 2])
network.create_structure()
# network.test_structure()


# X = [[1, 2, 3, 4, 5],
#      [5, 2, 7, 1, 7],
#      [4, 7, 33, 5, 7],
#      [2, 5, 7, 4, 8]]
#
# layer_one = Layer(5, 1)
# layer_one.forward(X)
# print(layer_one.output)
