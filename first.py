import numpy as np
import time
import random
import cv2


# import matplotlib.pyplot as plt
# from PIL import Image


def sigmoid(z):
    """Activation function"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, layers_sizes_array):
        print("---------------\nInitialized network object")
        # time.sleep(1)
        self.num_layers = len(layers_sizes_array)
        print("Set num_layers to: {0}".format(self.num_layers))
        # time.sleep(1)
        self.layers = layers_sizes_array
        print("Set layers to: {0}".format(self.layers))
        # time.sleep(1)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        print("\nSet biases to a list of numpy arrays with length: {0}".format(len(self.biases)))
        # time.sleep(1)
        for i, bias in enumerate(self.biases):
            print("{0}st array in list of biases is vector with shape(rows,columns): {1}".format(i, bias.shape))
            # time.sleep(1)

        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        print("\nSet weights to a list of numpy 2D arrays with length: {0}".format(len(self.weights)))
        # time.sleep(1)
        for i, weight in enumerate(self.weights):
            print("{0}st array in list of weights is matrix with shape(rows,columns): {1}, first number symbolising "
                  "number of neurons, and second symbolising number of inputs to each neuron".format(i, weight.shape))
            # time.sleep(1)
        print("---------------")

    def feedforward(self, a):
        """Return the output of the network if 'a' is an input"""
        # print("---------------\n Feedforward activated")
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print("{0}st weights matrix 'w' is \n{1}".format(i, w))
            # time.sleep(2)
            # print("{0}st input vector 'a' is\n {1}".format(i, a))
            # time.sleep(2)
            # print("{0}st biases vector 'b' is\n {1}".format(i, b))
            # time.sleep(2)
            a = sigmoid(np.dot(w, a) + b)
            # print("{0}st output and also next layer input is\n {1}".format(i, a))
            # time.sleep(2)
        # print("---------------")
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        pass


network = Network([5, 4, 3, 2])
network.feedforward(np.random.randn(5, 1))












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

# network = Network([9, 5, 3, 2])
# network.create_structure()
# network.test_structure()


# X = [[1, 2, 3, 4, 5],
#      [5, 2, 7, 1, 7],
#      [4, 7, 33, 5, 7],
#      [2, 5, 7, 4, 8]]
#
# layer_one = Layer(5, 1)
# layer_one.forward(X)
# print(layer_one.output)
