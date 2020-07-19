import gzip
import pickle
from mlxtend.data import loadlocal_mnist
import numpy as np


def load_data():
    f = gzip.open('./assets/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()

    # print("File mnist.pkl.gz has file type {0}".format(type(f)))
    # print("It contains three tuples 'training_data' 'validation_data' and 'test_data'")
    # print("Each tuple contains two values: first is array of images and second is array of numbers represented by these images".format())
    # print("Images (first tuple entry) is in form of {0} with shape(rows, columns) {1}".format(type(training_data[0]), training_data[0].shape))
    # print("This means, each of 50000 rows is an array of 784 numbers, representing 784 pixels forming greyscale image of a number")
    # print("Numbers (second tuple entry) is in form of {0} with shape(rows, columns) {1}".format(type(training_data[1]), training_data[1].shape))
    # print("This means second tuple entry is a vector, with each entry representing number shown on image")

    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():

    """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""

    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
