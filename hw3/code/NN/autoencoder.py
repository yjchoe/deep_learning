"""
Autoencoder models
"""

from __future__ import division
import numpy as np
import cPickle as pkl
import os

from scipy.special import expit

from nn.layer import AEInputLayer, AEOutputLayer
from nn.activation import Activation
from nn.algorithm import LearningRate
from nn.utils import generate_batches

class Autoencoder(object):
    """An autoencoder class.

    A simple autoencoder with binary input, binary hidden units, and 
    tied weights. Training is done using gradient descent.

    Attributes:
        n_visible: int
            Number of visible (input) units.
        n_hidden: int
            Number of hidden units.
        binary: bool
            Whether or not the input units are binary.
        denoising: float (between 0 and 1)
            Dropout rate for denoising autoencoders. 
            Using 0.0 (default) gives simple autoencoders without denoising.
        learning_rate: float or .nn.algorithm.LearningRate
            A function that takes the current epoch number and returns 
            a learning rate. `float` input becomes the `const` term.
            Defaults to `lambda epoch: const / (epoch // 100 + 1.)`.
        momentum: float (between 0 and 1)
            Momentum parameter for exponential averaging of previous 
            gradients.
        weight_decay: float
            A weight decay / L2 regularization parameter. Defaults to `1e-4`.
        early_stopping: boolean
            If true (default), attempts to stop training 
            before the validation error starts to increase.
        seed: float
            Random seed for initialization and sampling.

    Non-input Attributes:
        W: numpy.ndarray
            The weight matrix of shape `(n_hidden, n_visible)`.
        b: numpy.ndarray
            The bias vector for hidden units of length `n_hidden`.
        c: numpy.ndarray
            The bias vector for visible units of length `n_visible`.
        rng: numpy.random.RandomState
            NumPy random number generator using `seed`.

    Methods:
        __init__, save, load, train, 
        reconstruct, compute_error

    """

    def __init__(self, n_visible=784, n_hidden=100, binary=True,
                 activation='sigmoid', denoising=0.0,
                 learning_rate=0.1, momentum=0.5, weight_decay=1e-4, 
                 early_stopping=True, seed=99):
        """
        Autoencoder model initializer.
        """

        # Attributes
        self.n_visible      = n_visible
        self.n_hidden       = n_hidden
        self.binary         = binary
        self.activation     = activation
        self.denoising      = denoising
        self.learning_rate  = learning_rate
        self.momentum       = momentum
        self.weight_decay   = weight_decay
        self.early_stopping = early_stopping
        self.seed           = seed

        # Turn `activation` and `learning_rate` to class instances
        if not isinstance(self.activation, Activation):
            self.activation = Activation(self.activation)
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Denoising rate
        assert 0.0 <= self.denoising <= 1.0

        # Input layer
        encoder = AEInputLayer('AE_input_layer', n_visible, n_hidden, 
                               self.activation, 
                               self.learning_rate, self.momentum, 
                               self.weight_decay, 0.0, self.seed+1)
        # Pointers to randomly initialized weights and biases
        self.W = encoder.W
        self.b = encoder.b

        # Output layer with tied weights (bias is different: `self.c`)
        decoder = AEOutputLayer('AE_output_layer', n_hidden, n_visible, 
                                self.W, self.activation,
                                self.learning_rate, self.momentum,
                                self.weight_decay, 0.0, self.seed+2)
        self.c = decoder.b
        self.layers = [encoder, decoder]

        # RNG for denoising
        self.rng = np.random.RandomState(self.seed)

        # Training updates
        self.epoch = 0
        self.training_error = []
        self.validation_error = []
    
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
        rbm = pkl.load(fname)
        if isinstance(rbm, RBM):
            return rbm
        else:
            raise Exception('Loaded object is not a `RBM` object.')

    def train(self, X, X_valid=None, 
              batch_size=200, n_epoch=40, batch_seed=127, verbose=True):
        """Train the autoencoder using backpropagation.

        Args:
            X: numpy.ndarray (binary)
                Input data of size `n` (sample size) by `p` (data dimension).
            X_valid: numpy.ndarray (binary)
                Optional validation data matrix. If provided,
                current validation error is stored in the model.
            batch_size: int
                Size of random batches of the input data.
            n_epoch: int
                Number of epochs to train on the input data.
            batch_seed: int
                First random seed for batch selection.
            verbose: bool
                If true (default), report training updates per epoch to stdout.
        Returns:
            rbm: RBM
                Trained `RBM` model.
        """

        assert self.W.shape[1] == X.shape[1]
        if X_valid is not None:
            assert X.shape[1] == X_valid.shape[1]
        elif self.early_stopping:
            raise Exception('RBM.train: no validation data for early stopping.')
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |       Cross-Entropy       |       Cross-Entropy       |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for i, batch in enumerate(\
                generate_batches(n, batch_size, batch_seed + t)):

                # Forward propagation (last h is output prob)
                if self.denoising > 0.0:
                    x = X[batch, :]
                    mask = self.rng.binomial(1, 1.-self.denoising, size=x.shape)
                    h = x * mask
                else:
                    h = X[batch, :]
                for l in self.layers:
                    h = l.fprop(h, update_units=True)

                # Backpropagation
                grad   = -(X[batch, :] - h)  # cross-entropy or squared loss
                for l in self.layers[::-1]:
                    grad = l.bprop(grad)

                # Update tied weight gradients
                lr = self.learning_rate.get()
                for l in self.layers[::-1]:
                    self.W = self.W - lr * (l.grad_W + l.grad_decay)
                # Reassign updated tied weights
                for l in self.layers:
                    l.W = self.W

                # Update biases (SGD already performed in each layer)
                self.b = self.layers[0].b
                self.c = self.layers[1].b

            self.epoch += 1
            self.learning_rate.epoch = self.epoch
            for l in self.layers:
                l.learning_rate.epoch = self.epoch

            # Cross-entropy error based on stochastic reconstructions 
            # using updated parameters
            training_error = self.compute_error(X)
            self.training_error.append((self.epoch, training_error))

            if X_valid is not None:
                validation_error = self.compute_error(X_valid)
                self.validation_error.append((self.epoch, validation_error))
                if verbose:
                    print('|  {:3d}  |         {:9.5f}         |         {:9.5f}         |'.\
                        format(self.epoch, training_error, validation_error))
                if self.early_stopping:
                    if (self.epoch >= 100 and
                        1.02 * min(self.validation_error[-2][1],
                                   self.validation_error[-3][1],
                                   self.validation_error[-4][1],
                                   self.validation_error[-5][1],
                                   self.validation_error[-6][1]) < validation_error):
                        print('======Early stopping: validation error increase at epoch {:3d}====='.\
                            format(self.epoch))
                        break
            else:
                if verbose:
                    print('|  {:3d}  |         {:9.5f}         |                           |'.\
                        format(self.epoch, training_error))

        if verbose:
            print('|-------|---------------------------|---------------------------|')

        return self

    def reconstruct(self, X, prob=False):
        """Reconstruct `X` using the autoencoder (forward propagation).

        Args:
            X: numpy.ndarray 
                Input data of size `n` (sample size) by 
                `n_visible` (input data dimension).
            prob: bool
                This only applies when `self.binary == True`.
                If `True`, returns the sigmoid probabilities for each
                dimension. If `False` (default), returns the thresholded
                binary reconstruction.

        Returns:
            X_hat: numpy.ndarray
                Reconstructed samples corresponding to each data point in `X`.
                Size `n` (sample size) by `n_visible` (input data dimension).
        """
        assert self.n_visible == X.shape[1]

        h = X
        for l in self.layers:
            h = l.fprop(h, update_units=False)

        if self.binary and not prob:
            return (h >= 0.5).astype(np.int8)
        else:
            return h

    def compute_error(self, X):
        """Computes the error (cross-entropy if binary; squared loss if real)
        between `X` and its reconstruction under the current model.

        This gives the primary evaluation metric for autoencoders.
        Note that the error is summed over the number of visible units and 
        is scaled by the sample size (i.e. divided by `n`). This allows us
        to compare training and validation errors on the same scale.

        Args:
            X: numpy.ndarray
                Input data to be predicted. 
                Size `n` (sample size) by `n_visible` (data dimension).
        Returns:
            error: float
                Mean cross-entropy error of stochastic reconstruction.
        """

        X_hat = self.reconstruct(X, prob=True)
        if self.binary:
            return -(X * np.log(X_hat + 1e-8) +
                     (1 - X) * np.log((1 - X_hat) + 1e-8)).sum(axis=1).mean()
        else:
            return ((X - X_hat)**2).sum(axis=1).mean()
