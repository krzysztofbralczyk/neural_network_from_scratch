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
        # print("---------------\nInitialized network object")
        # time.sleep(3)
        self.num_layers = len(layers_sizes_array)
        # print("Amount of layers: {0}".format(self.num_layers))
        # time.sleep(3)
        self.layers = layers_sizes_array
        # print("Layers sizes: {0}".format(self.layers))
        # time.sleep(3)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        # print("\nBiases: a list of numpy arrays with length: {0}".format(len(self.biases)))
        # time.sleep(3)
        # for i, bias in enumerate(self.biases):
        #     print("\t{0}st array in list of biases is vector with shape(rows,columns): {1}".format(i, bias.shape))
        #     time.sleep(3)

        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        # print("\nWeights: a list of numpy 2D arrays with length: {0}".format(len(self.weights)))
        # time.sleep(3)
        # for i, weight in enumerate(self.weights):
        #     print("\t{0}st array in list of weights is matrix with shape(rows,columns): {1}, first number symbolising "
        #           "number of neurons, and second symbolising number of inputs to each neuron".format(i, weight.shape))
        #     time.sleep(3)
        # print("\nNetwork initialization complete\n")
        # time.sleep(3)

    def feedforward(self, a):
        """Return the output of the network if 'a' is an input"""
        # print("---------------\n Feedforward activated")
        # time.sleep(3)
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print("{0}st weights matrix 'w' is \n{1}".format(i, w))
            # time.sleep(3)
            # print("{0}st input vector 'a' is\n {1}".format(i, a))
            # time.sleep(3)
            # print("{0}st biases vector 'b' is\n {1}".format(i, b))
            # time.sleep(3)
            a = sigmoid(np.dot(w, a) + b)
        #     print("{0}st output and also next layer input is\n {1}".format(i, a))
        #     time.sleep(3)
        # print("---------------")
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # print("---------------------\nStart training neural network")
        # time.sleep(3)
        # print("Training data is expected to be in form of list of tuples (X,y)")
        # time.sleep(3)
        # print("X is input and y is desired output")
        # time.sleep(3)
        # print("X is nd array vector (784 X 1) and y is nd array vector (10 X 1)")
        # time.sleep(3)

        if test_data: n_test = len(test_data)
        n = len(training_data)
        # print("Length of training data is: {0} tuples".format(n))
        # time.sleep(3)
        # if test_data: print("Test data has been provided. Length of test data is: {0} tuples".format(n_test))
        # else: print("No test data provided")
        # time.sleep(3)
        # print("\nCommencing learning")
        # time.sleep(3)
        for j in range(epochs):
            # print("\n\tTraining data randomly shuffled")
            # time.sleep(3)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # print("\tData separated into mini batches {0} mini "
            #       "batches. Each has size {1}".format(len(mini_batches), mini_batch_size))
            # time.sleep(4)
            # print("\tStart actual learning (changing weights and biases)\n")
            # time.sleep(4)
            for i, mini_batch in enumerate(mini_batches):
                # print("\t\t {0}st mini batch provided".format(i))
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                # print("Because test data was provided, initialize testing.")
                # time.sleep(3)
                # print("Result of neural network after training for {0} iterations is: ".format(j))
                # time.sleep(3)
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                # time.sleep(3)
            else:
                print("Epoch {0} complete".format(j))
                # time.sleep(3)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # print("\t\tStarting backpropagation algorithm for each training input of mini batch")
        # time.sleep(4)
        for i, (x, y) in enumerate(mini_batch):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # print("\t\tAdded influence of {0}st training input to gradient's lists".format(i))
            # time.sleep(3)

        # print("\t\tAveraging gradient's values by size of batch")
        # time.sleep(3)
        # print("\t\tUpdated network's weights and biases based on averaged gradient")
        # time.sleep(3)
        self.weights = [w - nw * (eta/len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - nb * (eta/len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        # print("\t\tFinished learning from given mini batch\n")
        # time.sleep(3)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x

        # print("\t\t\tTraining data sent to first layer.")
        # time.sleep(2)
        # print("\t\t\tCommencing feedforward.")
        # time.sleep(3)

        activations = [x]
        zs = []

        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print("\n\t\t\tCalculating {0}st layer".format(i+1))
            # time.sleep(3)
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            # print("\t\t\tCalculated activations vector.")
            # time.sleep(2)
            # print("\t\t\tOutput of {0}st layer sent to next layer.".format(i+1))
            # time.sleep(3)
            activations.append(activation)
        # print("\t\t\tAll weighted inputs and activations saved in two lists for future calculations")
        # time.sleep(3)
        # print("\t\t\tFeedforward ended\n")
        # time.sleep(3)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # print("\t\t\tCalculating vector of errors for last layer")
        # time.sleep(3)
        # print("\t\t\tVector of error shows how layer output should be changed to minimize cost function")
        # time.sleep(3)
        # print("\t\t\tError vector is:")
        # time.sleep(3)
        # print("\n", delta, "\n")
        # I give up here, this shit is rocket science xD
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


import mnistData_testing as mnist

training_data, validation_data, test_data = mnist.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0)
print(net.feedforward(training_data[0][0]))













# network = Network([5, 4, 3, 2])
# network.feedforward(np.random.randn(5, 1))
# network.SGD([(1, 2), (3, 4), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6), (5, 6)], 1, 10, 3.0)





















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
