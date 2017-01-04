"""
Feedforward neural network models
"""

from __future__ import division
import numpy as np
import cPickle as pkl
import os

from layer import HiddenLayer, OutputLayer
from activation import Activation
from algorithm import LearningRate
from utils import transform_y, generate_batches

class NN(object):
    """A neural network class.

    Fully-connected feedforward single- or multi-layer neural networks 
    with nonlinear activation functions, weight decay and dropouts. 
    Output layer is assumed to be a softmax function.

    Attributes:
        architecture: list of ints
            A list of integers that contain the number of neurons per layer.
            For example, `(784, 100, 100, 10)` indicates a two-hidden-layer NN
            with 784-dimensional inputs, 100 hidden neurons in each of the 
            two hidden layers, and a 10-dimensional output.
        activation: string or Activation
            Choice of an activation function. 
            Either 'sigmoid' (default), 'tanh', 'relu', or 'linear', or
            the corresponding instance of the `Activation` class.
        learning_rate: function(epoch) or float
            A function that takes the current epoch number and returns 
            a learning rate. Defaults to `lambda epoch: .1 / (epoch % 200 + 1.)`.
        momentum: float (between 0 and 1)
            Momentum parameter for exponential averaging of previous 
            gradients.
        weight_decay: float
            A weight decay / L2 regularization parameter. Defaults to `1e-4`.
        dropout: float (between 0 and 1)
            Probability of a unit getting dropped out. Defaults to `0.5`.
        early_stopping: boolean
            If true (default), attempts to stop training 
            before the validation error starts to increase.
        seed: float
            Random seed for initialization and dropout.

    Non-input Attributes:
        layers: list of `Layer`s
            A list of layers that make up the neural network. A layer includes
            the incoming weight matrix and the bias vector to its units.
            The first layer connects the input to the first set of hidden 
            units, while the last layer connects the last set of hidden units 
            to the output probabilities (i.e. softmax).


    Methods:
        __init__, save, load, train, 
        predict, compute_error, compute_cross_entropy

    """

    def __init__(self, architecture=[784, 100, 10], 
                 activation='sigmoid', learning_rate=0.1, momentum=0.5,
                 weight_decay=1e-4, dropout=0.5, early_stopping=True, seed=99):
        """
        Neural network model initializer.
        """

        # Attributes
        self.architecture   = architecture
        self.activation     = activation
        self.learning_rate  = learning_rate
        self.momentum       = momentum
        self.weight_decay   = weight_decay
        self.dropout        = dropout
        self.early_stopping = early_stopping
        self.seed           = seed

        # Turn `activation` and `learning_rate` to class instances
        if not isinstance(self.activation, Activation):
            self.activation = Activation(self.activation)
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Initialize a list of layers
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(architecture[:-2], 
                                              architecture[1:-1])):
            l = HiddenLayer('layer{}'.format(i), n_in, n_out, self.activation, 
                            self.learning_rate, self.momentum, 
                            self.weight_decay, self.dropout, self.seed+i)
            self.layers.append(l)
        # Output layer
        n_in, n_out = architecture[-2], architecture[-1]
        l = OutputLayer('output_layer', n_in, n_out, 
                        self.learning_rate, self.momentum,
                        self.weight_decay, self.dropout, self.seed+i+1)
        self.layers.append(l)

        # Training updates
        self.epoch = 0
        self.training_error = []
        self.validation_error = []
        self.training_loss = []
        self.validation_loss = []
    
    def save(self, path):
        """
        Save the current model in `path`.
        """
        with open(path, 'w') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load a model saved by the function `save`.
        """
        with open(path) as f:
            rbm = pkl.load(f)
        if isinstance(nn, NN):
            return nn
        else:
            raise Exception('Loaded object is not a `NN` object.')

    def train(self, X, y, X_valid=None, y_valid=None,
              batch_size=200, n_epoch=40, batch_seed=0, verbose=True):
        """Train the neural network with data.

        Args:
            X: numpy.ndarray
                Input data of size `n` (sample size) by `p` (data dimension).
            y: numpy.ndarray (binary)
                One-hot labels of size `n` (sample size) by `k` (# classes).
            X_valid: numpy.ndarray
                Optional validation data matrix. If provided with `y_valid`,
                current validation error rate is stored in the model.
            y_valid: numpy.ndarray
                Optional validation outcome vector. If provided with `X_valid`,
                current validation error rate is stored in the model.
            batch_size: int
                Size of random batches of the input data.
            n_epoch: int
                Number of epochs to train on the input data.
            batch_seed: int
                First random seed for batch selection.
            verbose: bool
                If true (default), report training updates per epoch to stdout.
        Returns:
            nn: NN
                Trained `NN` model.
        """

        assert self.layers[0].n_in == X.shape[1]
        assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |    Error    |  Cross-Ent  |    Error    |  Cross-Ent  |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for i, batch in enumerate(\
                generate_batches(n, batch_size, batch_seed + t)):

                # Forward propagation (last h is output prob)
                h = X[batch, :]
                for l in self.layers:
                    h = l.fprop(h, update_units=True)

                # Backpropagation
                grad = -(y[batch, :] - h)
                for l in self.layers[::-1]:
                    grad = l.bprop(grad)

            self.epoch += 1
            for l in self.layers:
                l.learning_rate.epoch = self.epoch

            # Errors
            training_error = self.compute_error(X, y)
            training_loss  = self.compute_cross_entropy(X, y)
            self.training_error.append((self.epoch, training_error))
            self.training_loss .append((self.epoch, training_loss ))

            if X_valid is not None and y_valid is not None:
                validation_error = self.compute_error(X_valid, y_valid)
                validation_loss  = self.compute_cross_entropy(X_valid, y_valid)
                self.validation_error.append((self.epoch, validation_error))
                self.validation_loss .append((self.epoch, validation_loss))
                if verbose:
                    print('|  {:3d}  |   {:.5f}   |   {:.5f}   |   {:.5f}   |   {:.5f}   |'.\
                        format(self.epoch, training_error, training_loss, 
                               validation_error, validation_loss))
                if self.early_stopping:
                    if (self.epoch >= 40 and
                        self.validation_loss[-2][1] < validation_loss and
                        self.validation_loss[-3][1] < validation_loss and
                        self.validation_loss[-4][1] < validation_loss):
                        print('======Early stopping: validation loss increase at epoch {:3d}======'.\
                            format(self.epoch))
                        break
            else:
                if verbose:
                    print('|  {:3d}  |   {:.5f}   |   {:.5f}   |             |             |'.\
                        format(self.epoch, training_error, training_loss))

        if verbose:
            print('|-------|---------------------------|---------------------------|')

        return self


    def predict(self, X, output_type='response'):
        """Predict labels using current model parameters.

        Args:
            X: numpy.ndarray
                Input data to be predicted. 
                Size `n` (sample size) by `p` (data dimension).
            output_type: string
                Type of output. 
                'response' (default) returns predicted `y` as one-hot vectors.
                'prob' or 'probability' returns the softmax probability.
        Returns:
            One of the following:
                y: numpy.ndarray (binary)
                    Predicted labels in one-hot format, 
                    if `output_type` is 'response'. (default)
                    Size `n` (sample size) by `c` (number of classes).
                p: numpy.ndarray
                    Predicted softmax probabilities, 
                    if `output_type` is 'prob'` or 'probability'.
                    Size `n` (sample size) by `c` (number of classes).
        """

        assert self.layers[0].n_in == X.shape[1]

        h = X
        for l in self.layers:
            h = l.fprop(h)

        if output_type == 'response':
            return transform_y(np.argmax(h, axis=1), h.shape[1])
        elif output_type == 'prob' or output_type == 'probability':
            return h
        else:
            raise Exception('NN.predict: unrecognized `output_type`')

    def compute_error(self, X, y):
        """Computes error rate on `X` and `y`.

        Args:
            X: numpy.ndarray
                Input data to be predicted. 
                Size `n` (sample size) by `p` (data dimension).
            y: numpy.ndarray (binary)
                Labels in one-hot format.
                Size `n` (sample size) by `c` (number of classes).
        Returns:
            err: float
                Prediction error rate.
        """
        assert X.shape[0] == y.shape[0]
        return 1. - np.all(self.predict(X) == y, axis=1).mean()

    def compute_cross_entropy(self, X, y):
        """Computes the cross-entropy loss (negative log-likelihood) 
        between `y` and `self.predict(X)`.

        Args:
            X: numpy.ndarray
                Input data to be predicted. 
                Size `n` (sample size) by `p` (data dimension).
            y: numpy.ndarray (binary)
                Labels in one-hot format.
                Size `n` (sample size) by `c` (number of classes).
        Returns:
            loss: float
                Mean cross-entropy loss over data.
        """

        assert X.shape[0] == y.shape[0]
        p = self.predict(X, 'prob')

        return -(y * np.log(p+1e-8)).mean()
