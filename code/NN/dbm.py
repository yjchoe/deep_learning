"""
Deep Boltzmann machines
"""

from __future__ import division
import numpy as np
import cPickle as pkl
import os

from scipy.special import expit

from nn.algorithm import LearningRate
from nn.utils import generate_batches

class DBM(object):
    """A 2-layer deep Boltzmann machine (DBM) class.

    2-layer DBM with binary input and hidden units. 
    Inference is done using mean-field variation inference and
    persistent contrastive divergence (CD).

    Attributes:
        n_visible: int
            Number of visible (input) units.
        n_hidden1: int
            Number of hidden units in the first hidden layer.
        n_hidden2: int
            Number of hidden units in the second hidden layer.        
        n_chains: int
            Number of persistent Gibbs chains.
            Defaults to 100.
        n_vi_steps: int
            Number of mean-field variational approximation steps.
            Defaults to 10.
        n_gibbs_steps: int
            Number of Gibbs sampling steps in each CD chain.
            Defaults to 1.
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
        generate_negative_sample, run_gibbs,
        sample_hidden1, sample_hidden2, sample_visible, 
        compute_cross_entropy

    """

    def __init__(self, n_visible=784, n_hidden1=100, n_hidden2=100,
                 n_chains=100, n_vi_steps=10, n_gibbs_steps=1,
                 learning_rate=0.01, early_stopping=True, seed=99):
        """
        DBM model initializer.
        """

        # Attributes
        self.n_visible      = n_visible
        self.n_hidden1      = n_hidden1
        self.n_hidden2      = n_hidden2
        self.n_chains       = n_chains
        self.n_vi_steps     = n_vi_steps
        self.n_gibbs_steps  = n_gibbs_steps
        self.learning_rate  = learning_rate
        self.early_stopping = early_stopping
        self.seed           = seed

        # Turn `learning_rate` to class instances
        if not isinstance(self.learning_rate, LearningRate):
            self.learning_rate = LearningRate(self.learning_rate)

        # Initialize weights (an analogous strategy from feedforward NN)
        self.rng = np.random.RandomState(seed)
        u = np.sqrt(6. / (self.n_visible + self.n_hidden1))
        self.W1 = self.rng.uniform(-u, u, 
                                   size=(self.n_hidden1, self.n_visible))
        self.b1 = np.zeros((self.n_hidden1, 1))
        u = np.sqrt(6. / (n_hidden1 + n_hidden2))
        self.W2 = self.rng.uniform(-u, u, 
                                   size=(self.n_hidden2, self.n_hidden1))
        self.b2 = np.zeros((self.n_hidden2, 1))
        self.c = np.zeros((self.n_visible, 1))

        # Initialize mean-field parameters
        self.mu1 = self.rng.uniform(0, 1, size=(n_hidden1, 1))
        self.mu2 = self.rng.uniform(0, 1, size=(n_hidden2, 1))

        # Initialize persistent chains
        self.X_chain  = self.rng.binomial(1, 0.5, 
                                          size=(self.n_chains, self.n_visible))
        self.H1_chain = self.rng.binomial(1, 0.5, 
                                          size=(self.n_chains, self.n_hidden1))
        self.H2_chain = self.rng.binomial(1, 0.5, 
                                          size=(self.n_chains, self.n_hidden2))

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
    def load(path):
        """
        Load a model saved by the function `save`.
        """
        with open(path) as f:
            dbm = pkl.load(f)
        if isinstance(dbm, DBM):
            return dbm
        else:
            raise Exception('Loaded object is not a `DBM` object.')

    def train(self, X, X_valid=None, 
              batch_size=10, n_epoch=50, batch_seed=127, verbose=True):
        """Train the DBM using mean-field variational inference and
        persistent contrastive divergence.

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
            dbm: DBM
                Trained `DBM` model.
        """

        assert self.W1.shape[1] == X.shape[1]
        if X_valid is not None:
            assert X.shape[1] == X_valid.shape[1]
        elif self.early_stopping:
            raise Exception('DBM.train: no validation data for early stopping.')
        n = X.shape[0]
        n_batches = int(np.ceil(n / batch_size))

        if verbose:
            print('|-------|---------------------------|---------------------------|')
            print('| Epoch |         Training          |         Validation        |')
            print('|-------|---------------------------|---------------------------|')
            print('|   #   |       Cross-Entropy       |       Cross-Entropy       |')
            print('|-------|---------------------------|---------------------------|')

        for t in range(n_epoch):

            for batch in generate_batches(n, batch_size, batch_seed + t):

                # Get X in the current batch 
                X_batch = X[batch, :]
                n_batch = X_batch.shape[0]

                # Mean-field variational inference
                self.mu1 = self.rng.uniform(0, 1, 
                                            size=(n_batch, self.n_hidden1))
                self.mu2 = self.rng.uniform(0, 1, 
                                            size=(n_batch, self.n_hidden2))
                for i in range(len(batch)):
                    for _ in range(self.n_vi_steps):
                        self.mu1[i] = expit(self.W1.dot(X_batch[i]) +
                                            self.W2.T.dot(self.mu2[i]) + 
                                            self.b1.T)
                        self.mu2[i] = expit(self.W2.dot(self.mu1[i]) + 
                                            self.b2.T)

                # Data-dependent expectations (positive phase)
                grad_W1_data = (1. / n_batch) * self.mu1.T.dot(X_batch)
                grad_W2_data = (1. / n_batch) * self.mu2.T.dot(self.mu1)
                grad_b1_data = self.mu1.mean(axis=0, keepdims=True).T
                grad_b2_data = self.mu2.mean(axis=0, keepdims=True).T
                grad_c_data  = X_batch.mean(axis=0, keepdims=True).T


                # Persistent contrastive divergence via Gibbs sampling
                self.X_chain, self.H1_chain, self.H2_chain = \
                    self.run_gibbs(self.X_chain, self.H2_chain)

                # Data-independent model expectations (negative phase)
                grad_W1_model = (1. / self.n_chains) * \
                                    self.H1_chain.T.dot(self.X_chain)
                grad_W2_model = (1. / self.n_chains) * \
                                    self.H2_chain.T.dot(self.H1_chain)
                grad_b1_model = self.H1_chain.mean(axis=0, keepdims=True).T
                grad_b2_model = self.H2_chain.mean(axis=0, keepdims=True).T
                grad_c_model  = self.X_chain.mean(axis=0, keepdims=True).T

                # Perform contrastive divergence gradient updates
                lr = self.learning_rate.get()
                self.W1 += lr * (grad_W1_data - grad_W1_model)
                self.W2 += lr * (grad_W2_data - grad_W2_model)
                self.b1 += lr * (grad_b1_data - grad_b1_model)
                self.b2 += lr * (grad_b2_data - grad_b2_model)
                self.c  += lr * (grad_c_data  - grad_c_model )

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
                    if (self.epoch >= 40 and
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
        """Runs Gibbs sampling and only returns the new samples.
        """
        X_new, _, _ = self.run_gibbs(X)
        return X_new

    def run_gibbs(self, X, H2=None):
        """Generate Gibbs samples (x, h1, h2) corresponding to 
        each data point, starting from inputs (x, h2), 
        using `n_gibbs_steps` of contrastive divergence via Gibbs sampling.

        Note that intermediate Gibbs samples are not stored as they are 
        considered as burn-in samples.

        Args:
            X: numpy.ndarray (binary)
                Input data of size `n` (sample size) by 
                `n_visible` (input data dimension).
            H2: numpy.ndarray (binary, optional)
                Second-layer hidden units of size `n` (sample size) by
                `n_hidden2` (number of second-layer hidden units).
                If `None`, the hidden units are randomly initialized. 

        Returns:
            X_new: numpy.ndarray (binary)
                Negative samples corresponding to each data point in `X`.
                Size `n` (sample size) by `n_visible` (input data dimension).
            H1_new: numpy.ndarray (binary)
                Negative samples corresponding to each data point in `X`.
                Size `n` (sample size) by `n_hidden1` (first-layer dimension).
            H2_new: numpy.ndarray (binary)
                Negative samples corresponding to each data point in `X`.
                Size `n` (sample size) by `n_hidden2` (second-layer dimension).
        """
        assert self.n_visible == X.shape[1]
        X_new = X.copy()

        if H2 is not None:
            assert self.n_hidden2 == H2.shape[1]
            assert X.shape[0] == H2.shape[0]
            H2_new = H2.copy()
        else:
            H2_new = self.rng.binomial(1, 0.5, size=(X.shape[0], self.n_hidden2))

        for _ in range(self.n_gibbs_steps):
            H1_new = self.sample_hidden1(X_new, H2_new)
            H2_new = self.sample_hidden2(H1_new)
            X_new = self.sample_visible(H1_new)

        return X_new, H1_new, H2_new

    def sample_hidden1(self, X, H2, prob=False):
        """Sample first-layer hidden units given 
        visible units and second-layer hidden units.

        Can alternatively return the sigmoid probabilities.

        Args:
            X: numpy.ndarray (binary)
                Visible units of size `n` (sample size) by 
                `n_visible` (number of visible units).
            H2: numpy.ndarray (binary)
                First-layer hidden units of size `n` (sample size) by 
                `n_hidden2` (number of second-layer hidden units).
            prob: boolean
                If `True`, returns the sigmoid probability from which
                samples are generated instead of samples.

        Returns:
            numpy.ndarray
                Size `n` by `n_hidden1` (number of first-layer hidden units). 
                Either `n` samples or `n` probabilities.
        """
        assert self.n_visible == X.shape[1]
        assert self.n_hidden2 == H2.shape[1]
        assert X.shape[0] == H2.shape[0]

        P = expit(X.dot(self.W1.T) + H2.dot(self.W2) + self.b1.T)
        if prob:
            return P
        else:
            return self.rng.binomial(1, P)

    def sample_hidden2(self, H1, prob=False):
        """Sample second-layer hidden units given first-layer hidden units.

        Can alternatively return the sigmoid probabilities.
        Note that we do not need the visible units,
        which are independent of the second-layer hidden units given
        the first-layer hidden units.

        Args:
            H1: numpy.ndarray (binary)
                First-layer hidden units of size `n` (sample size) by 
                `n_hidden1` (number of first-layer hidden units).
            prob: boolean
                If `True`, returns the sigmoid probability from which
                samples are generated instead of samples.

        Returns:
            numpy.ndarray
                Size `n` by `n_hidden2` (input data dimension). 
                Either `n` samples or `n` probabilities.
        """
        assert self.n_hidden1 == H1.shape[1]

        P = expit(H1.dot(self.W2.T) + self.b2.T)
        if prob:
            return P
        else:
            return self.rng.binomial(1, P)

    def sample_visible(self, H1, prob=False):
        """Sample visible units given hidden units.

        Can alternatively return the sigmoid probabilities.
        Note that we do not need the second-layer hidden units,
        which are independent of the visible units given
        the first-layer hidden units.

        Args:
            H1: numpy.ndarray (binary)
                First-layer hidden units of size `n` (sample size) by 
                `n_hidden1` (number of first-layer hidden units).
            prob: boolean
                If `True`, returns the sigmoid probability from which
                samples are generated instead of samples.

        Returns:
            numpy.ndarray
                Size `n` by `n_visible` (input data dimension). 
                Either `n` samples or `n` probabilities.
        """
        assert self.n_hidden1 == H1.shape[1]

        P = expit(H1.dot(self.W1) + self.c.T)
        if prob:
            return P
        else:
            return self.rng.binomial(1, P)

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
        X_new, H1_new, H2_new = self.run_gibbs(X)
        P = self.sample_visible(H1_new, prob=True)  # not sampling!
        return -(X * np.log(P + 1e-8) + 
                 (1 - X) * np.log((1 - P) + 1e-8)).sum(axis=1).mean()
