"""
Restricted Boltzmann machines
"""

from __future__ import division
import numpy as np
import cPickle as pkl
import os

from scipy.special import expit

from nn.algorithm import LearningRate
from nn.utils import generate_batches

class RBM(object):
    """A restricted Boltzmann machine class.

    A simple RBM with binary input and hidden units. 
    Inference is performed using contrastive divergence (CD-k).
    **Updated:** Now supports persistent contrastive divergence.

    Attributes:
        n_visible: int
            Number of visible (input) units.
        n_hidden: int
            Number of hidden units.
        k: int
            Number of Gibbs sampling steps in the CD-k algorithm.
        persistent: bool
            If true (default), uses the previous negative sample as
            the initial visible units for the Gibbs chain. Otherwise,
            new Gibbs chain starts at the new batch of data.
        learning_rate: float or .nn.algorithm.LearningRate
            A function that takes the current epoch number and returns 
            a learning rate. `float` input becomes the `const` term.
            Defaults to `lambda epoch: const / (epoch // 100 + 1.)`.
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
        generate_negative_sample, 
        sample_from_posterior, sample_from_likelihood, 
        compute_posterior, compute_likelihood, compute_cross_entropy

    """

    def __init__(self, n_visible=784, n_hidden=100, k=1, persistent=True,
                 learning_rate=0.1, early_stopping=True, seed=99):
        """
        RBM model initializer.
        """

        # Attributes
        self.n_visible      = n_visible
        self.n_hidden       = n_hidden
        self.k              = k
        self.persistent     = persistent
        self.learning_rate  = learning_rate
        self.early_stopping = early_stopping
        self.seed           = seed

        # Turn `learning_rate` to class instances
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Initialize weights (an analogous strategy from feedforward NN)
        self.rng = np.random.RandomState(seed)
        u = np.sqrt(6. / (n_hidden + n_visible))
        self.W = self.rng.uniform(-u, u, size=(n_hidden, n_visible))
        self.b = np.zeros((n_hidden, 1))
        self.c = np.zeros((n_visible, 1))

        # Persistent chain initialization
        if persistent:
            self.X_neg = None

        # Training updates
        self.epoch = 0
        self.training_error = []
        self.validation_error = []
    
    def save(self, path):
        """
        Save the current model in `path`.
        """
        with open(path, 'w') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(path_dir):
        """
        Load a model saved by the function `save`.
        """
        with open(fname) as f:
            rbm = pkl.load(f)
        if isinstance(rbm, RBM):
            return rbm
        else:
            raise Exception('Loaded object is not a `RBM` object.')

    def train(self, X, X_valid=None, 
              batch_size=20, n_epoch=50, batch_seed=127, verbose=True):
        """Train the RBM using contrastive divergence (CD-k).

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

                # Get X in the current batch and its negative samples
                X_batch = X[batch, :]
                if persistent:
                    X_neg = self.generate_negative_sample(self.X_neg)
                else:
                    X_neg = self.generate_negative_sample(X_batch)
                n_batch = X_batch.shape[0]

                # Perform contrastive divergence gradient updates
                lr = self.learning_rate.get()
                p_batch = self.compute_posterior(X_batch)
                p_neg   = self.compute_posterior(X_neg)
                grad_W  = (1. / n_batch) * \
                            (p_batch.T.dot(X_batch) - p_neg.T.dot(X_neg))
                grad_b  = np.mean(p_batch - p_neg, axis=0, keepdims=True).T
                grad_c  = np.mean(X_batch - X_neg, axis=0, keepdims=True).T
                self.W += lr * grad_W
                self.b += lr * grad_b
                self.c += lr * grad_c

                if persistent:
                    self.X_neg = X_neg

            self.epoch += 1
            self.learning_rate.epoch = self.epoch

            # Cross-entropy error based on stochastic reconstructions 
            # using updated parameters
            training_error = self.compute_cross_entropy(X)
            self.training_error.append((self.epoch, training_error))

            if X_valid is not None:
                validation_error = self.compute_cross_entropy(X_valid)
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

    def generate_negative_sample(self, X):
        """Generate negative samples (\tlide{x}) corresponding to 
        each data point using `k` steps of Gibbs sampling.

        Note that intermediate Gibbs samples are not stored as they are 
        considered as burn-in samples.

        Args:
            X: numpy.ndarray (binary)
                Input data of size `n` (sample size) by 
                `n_visible` (input data dimension).

        Returns:
            X_neg: numpy.ndarray (binary)
                Negative samples corresponding to each data point in `X`.
                Size `n` (sample size) by `n_visible` (input data dimension).
        """
        assert self.n_visible == X.shape[1]

        X_new = X.copy()
        for _ in range(self.k):
            H_new = self.sample_from_posterior(X_new)
            X_new = self.sample_from_likelihood(H_new)

        return X_new

    def sample_from_posterior(self, X):
        """Sample from the posterior distribution given `X`, i.e.
                h_i ~ p ( h_i | x_i )
        for each i = 1, ..., n.

        Gives one sample from each row of `X`.
        Note that the independence assumption between hidden units is used.

        Args:
            X: numpy.ndarray (binary)
                Data-like matrix of size `n` (sample size) by 
                `n_visible` (input data dimension).

        Returns:
            H: numpy.ndarray (binary)
                Hidden variables of size `n` (sample size) by 
                `n_hidden` (number of hidden units).
        """
        P = self.compute_posterior(X)
        return self.rng.binomial(1, P)

    def sample_from_likelihood(self, H):
        """Sample from the likelihood function given `H`, i.e.
                x_i ~ p ( x_i | h_i )
        for each i = 1, ..., n.

        Gives one sample from each row of `H`.
        Note that the independence assumption between visible units is used.

        Args:
            H: numpy.ndarray (binary)
                Hidden variables of size `n` (sample size) by 
                `n_hidden` (number of hidden units).

        Returns:
            X: numpy.ndarray (binary)
                New sample of size `n` (sample size) by 
                `n_visible` (input data dimension).
        """
        L = self.compute_likelihood(H)
        return self.rng.binomial(1, L)

    def compute_posterior(self, X):
        """Compute the posterior probability 
                p ( h_j = 1 | x ) 
        conditioned on data `X`.

        This is the "upstream" inference in RBMs.

        Args:
            X: numpy.ndarray (binary)
                Input data of size `n` (sample size) by 
                `n_visible` (input data dimension).

        Returns:
            P: numpy.ndarray
                Posterior probabilities for each of `n` data points.
                Size `n` by `n_hidden` (number of hidden units). 
        """
        assert self.n_visible == X.shape[1]

        return expit(X.dot(self.W.T) + self.b.T)

    def compute_likelihood(self, H):
        """Compute the likelihood function 
                p ( x_k = 1 | h ) 
        of hidden units `H` (each row is a different `h`).

        This is the "downstream" inference in RBMs.

        Args:
            H: numpy.ndarray (binary)
                Hidden variables of size `n` (sample size) by 
                `n_hidden` (number of hidden units).

        Returns:
            L: numpy.ndarray
                Likelihood for each of `n` hidden unit vectors.
                Size `n` by `n_visible` (input data dimension). 
        """
        assert self.n_hidden == H.shape[1]

        return expit(H.dot(self.W) + self.c.T)

    def compute_cross_entropy(self, X):
        """Computes the cross-entropy error (negative log-likelihood) 
        between `X` and the likelihood of the current model for that data.

        This gives the primary evaluation metric for RBMs.
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

        assert self.n_visible == X.shape[1]
        H = self.sample_from_posterior(X)
        L = self.compute_likelihood(H)  # normalized between [0, 1]
        return -(X * np.log(L + 1e-8) + 
                 (1 - X) * np.log((1 - L) + 1e-8)).sum(axis=1).mean()
