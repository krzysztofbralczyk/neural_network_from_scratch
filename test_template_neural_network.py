import mnistData_testing as mnist
import template_neural_network as network

training_data, validation_data, test_data = mnist.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
