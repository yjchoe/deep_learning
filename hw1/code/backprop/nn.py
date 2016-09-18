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
            a learning rate. Defaults to `momentum(eps=0.01, beta=0.5)`.
            If a float `b` is provided, defaults to `lambda epoch: b`.
            Defaults to `0.1`.
        weight_decay: float
            A weight decay / L2 regularization parameter. Defaults to `1.0`.
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
        __init__, save, load, train, compute_error, predict

    """

    def __init__(self, architecture=[784, 100, 100, 10], 
                 activation='sigmoid', learning_rate=0.1, weight_decay=1.0, 
                 dropout=0.5, early_stopping=True, seed=0):
        """
        Neural network model initializer.
        """

        # Attributes
        self.architecture   = architecture
        self.activation     = activation
        self.learning_rate  = learning_rate
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
                            self.learning_rate, self.weight_decay, self.dropout, 
                            self.seed+i)
            self.layers.append(l)
        # Output layer
        n_in, n_out = architecture[-2], architecture[-1]
        l = OutputLayer('output_layer', n_in, n_out, 
                        self.learning_rate, self.weight_decay, self.dropout, 
                        self.seed+i+1)
        self.layers.append(l)

        # Training updates
        self.epoch = 0
        self.training_error = {}
        self.validation_error = {}
    
    def save(self, path):
        """
        Save the current model in `path`.
        """
        pkl.dump(self)

    @staticmethod
    def load(path_dir):
        """
        Load a model saved by the function `save`.
        """
        nn = pkl.load(fname)
        if isinstance(nn, NN):
            return nn
        else:
            raise Exception('Loaded object is not a `NN` object.')

    def train(self, X, y, X_valid=None, y_valid=None,
              batch_size=100, batch_seed=None):
        """
        Train the neural network with data.

        Args:
            X: numpy.ndarray
                Input data of size `n` (sample size) by `p` (data dimension).
            y: numpy.ndarray (binary)
                One-hot labels of size `n` (sample size) by `k` (number of classes).
            batch_size: int or None
                If provided (default 64), training is performed on each batch.
                Otherwise, training is done on the entire row of data.
            X_valid: numpy.ndarray
                Optional validation data matrix. If provided with `y_valid`,
                current validation error rate is stored in the model.
            y_valid: numpy.ndarray
                Optional validation outcome vector. If provided with `X_valid`,
                current validation error rate is stored in the model.
            batch_seed: float
                Random seed for batch selection.
        Returns:
            nn: NN
                Trained `NN` model.
        """

        assert self.layers[0].n_in == X.shape[1]
        assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        for i, batch in enumerate(generate_batches(n, batch_size, batch_seed)):

            # Forward propagation (first h is input; last h is output)
            h = X[batch, :]
            for l in self.layers:
                h = l.fprop(h, update_units=True)

            # Backpropagation
            grad = -(y[batch, :] - h)
            for l in self.layers[::-1]:
                grad = l.bprop(grad)

        self.epoch += 1
        print('Training Complete! ' + 
              '({:d} examples, epoch {:d})'.format(n, self.epoch))

        # Record and print training error
        training_error = self.compute_error(X, y)
        print('Training error: {:5f}'.format(training_error))
        self.training_error[self.epoch] = training_error
        # Record and print validation error (if data provided)
        if X_valid is not None and y_valid is not None:
            validation_error = self.compute_error(X_valid, y_valid)
            print('Validation error: {:.5f}'.format(validation_error))
            self.validation_error[self.epoch] = validation_error

        return self


    def predict(self, X):
        """
        Predict labels using current model parameters.

        Args:
            X: numpy.ndarray
                Input data to be predicted. 
                Size `n` (sample size) by `p` (data dimension).
        Returns:
            y: numpy.ndarray (binary)
                Predicted labels in one-hot format.
                Size `n` (sample size) by `c` (number of classes).
        """

        assert self.layers[0].n_in == X.shape[1]

        h = X
        for l in self.layers:
            h = l.fprop(h)

        return transform_y(np.argmax(h, axis=1), h.shape[1])

    def compute_error(self, X, y):
        """
        Computes error rate on `X` and `y`.

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
