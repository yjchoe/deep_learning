"""
Neural network layer class
"""

import numpy as np

class Layer(object):
    """A base class for layers of a feedforward neural network.

    Includes the weight matrix and the bias vector as well as 
    the hyperparameters for training. In the current implementation,
    this class is always instantiated by the `NN.__init__` method.

    Attributes:
        __name__: string
            Name of the layer such as `layer3`.
        n_in: int
            Number of input units to the layer.
        n_out: int
            Number of output units to the layer.
        learning_rate: function(epoch) or float
            A function that takes the current epoch number and returns 
            a learning rate. Defaults to `momentum(eps=0.01, beta=0.5)`.
            If a float `b` is provided, defaults to `lambda epoch: b`.
        weight_decay: float
            A weight decay / L2 regularization parameter. Defaults to `1`.
        dropout: float (between 0 and 1)
            Probability of a unit getting dropped out. Defaults to `0.5`.
        seed: float
            Random seed for initialization and dropout. 

    Non-input attributes:
        W: numpy.ndarray
            Weight matrix of size `n_out` times `n_in`.
        b: numpy.ndarray
            Bias vector of size `n_out`. 
        rng: numpy.random.RandomState
            Random number generator using `seed`.

    Methods:
        __init__, fprop, bprop
    """
    def __init__(self, name, n_in, n_out, 
                 learning_rate, weight_decay, dropout, seed): 
        """
        Neural network layer initializer.
        """

        # Attributes
        self.__name__      = name
        self.n_in          = n_in
        self.n_out         = n_out
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.dropout       = dropout
        self.seed          = seed

        # Weight and bias initialization
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.normal(size=(n_out, n_in)) / np.sqrt(n_in)
        self.b = np.zeros((n_out, 1))

        # Stored values for backpropagation
        self.h_in  = None
        self.a_out = None

    def fprop(self, h, update_units=False):
        """
        Forward propagation method. Implemented in the subclass. 
        """
        raise NotImplementedError()

    def bprop(self, grad):
        """
        Backpropagation method. Implemented in the subclass. 
        """
        raise NotImplementedError()


class HiddenLayer(Layer):
    """A subclass of `Layer` for the hidden layers.
    See subclass `OutputLayer` for the output layer construction. 

    Additional attributes:
        activation: Activation
            An instance of the Activation class.

    Additional non-input attributes:
        h_in, a_out: numpy.ndarray
            Intermediate unit values during backpropagation.

    Methods:
        __init__, fprop, bprop
    """

    def __init__(self, name, n_in, n_out, activation, 
                 learning_rate, weight_decay, dropout, seed): 
        """
        Neural network hidden layer initializer.
        """

        # Initialization from the superclass
        super(HiddenLayer, self).__init__(
            name, n_in, n_out, learning_rate, weight_decay, dropout, seed)

        # Additional attributes
        self.activation = activation
        self.h_in  = None
        self.a_out = None

    def fprop(self, h_in, update_units=False):
        """
        Forward propagation of incoming units through the current layer.
        Includes a linear transformation and an activation. 
        Can accept batch inputs of size `n`.

        Args:
            h_in: numpy.ndarray
                An `n` by `n_in` matrix that corresponds to 
                the hidden units from the immediate downstream. 
                Each row corresponds to the hidden unit value from
                each data point in the current batch.
            update_units: boolean
                If `True`, stores `h_in` and `a_in` that are later used for 
                backpropagation. The default `False` option is used for 
                prediction.
        Returns:
            h_out: numpy.ndarray
                An `n` by `n_out` matrix that corresponds to 
                the activated hidden units in the immediate upstream. 
                Each row corresponds to the hidden unit value from
                each data point in the current batch.
        """

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in
        
        # For each data point, this is `a_out = W.dot(h_in) + b`
        a_out = h_in.dot(self.W.T) + self.b.T

        if update_units:
            self.h_in  = h_in
            self.a_out = a_out

        return self.activation.eval(a_out)

    def bprop(self, grad_h_out):
        """
        Backpropagation of the gradient w.r.t. the *post-activation* units 
        in the upstream (output direction).
        Updates model parameters (`W` and `b`) and returns the gradient w.r.t.
        the post-activation units in the downstream.

        For both the argument and the return value, each row corresponds to 
        the gradient from each data point in the current batch.

        Args: 
            grad_h_out: numpy.ndarray
                An `n` by `n_out` matrix of gradients with respect to
                the *post-activation* units in the immediate upstream. 
        Returns:
            grad_h_in: numpy.ndarray
                An `n` by `n_in` matrix of gradients with respect to
                the post-activation units in the immediate downstream.
        """

        # Assert that `fprop` is already done
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0]  == grad_h_out.shape[0]
        assert self.a_out.shape[0] == grad_h_out.shape[0]
        n = self.h_in.shape[0]

        # Compute gradients
        grad_a_out = grad_h_out * self.activation.grad(self.a_out)
        grad_W     = (1. / n) * grad_a_out.T.dot(self.h_in)
        grad_decay = 2. * self.weight_decay * self.W
        grad_b     = grad_a_out.mean(axis=0, keepdims=True).T
        grad_h_in  = grad_a_out.dot(self.W)

        # SGD Updates
        lr = self.learning_rate.get()
        self.W = self.W - lr * (grad_W + grad_decay)
        self.b = self.b - lr * (grad_b)

        return grad_h_in

class OutputLayer(Layer):
    """A subclass of `Layer` for the softmax output layer.
    See subclass `HiddenLayer` for the hidden layer construction. 

    Additional non-input attributes:
        h_in: numpy.ndarray
            Last post-activation units before softmax. Used in backprop.

    Methods:
        __init__, fprop, bprop
    """

    def __init__(self, name, n_in, n_out, 
                 learning_rate, weight_decay, dropout, seed): 
        """
        Neural network output layer initializer.
        """

        # Initialization from the superclass
        super(OutputLayer, self).__init__(
            name, n_in, n_out, learning_rate, weight_decay, dropout, seed)

        # Additional attributes
        self.h_in = None


    def fprop(self, h_in, update_units=False):
        """
        Forward propagation of the last hidden units to the softmax layer.
        Includes a linear transformation and a softmax transformation.
        Can accept batch inputs of size `n`.

        Args:
            h_in: numpy.ndarray
                An `n` by `n_in` matrix that corresponds to 
                the hidden units from the immediate downstream. 
                Each row corresponds to the hidden unit value from
                each data point in the current batch.
            update_units: boolean
                If `True`, stores `h_in` and `a_in` that are later used for 
                backpropagation. The default `False` option is used for 
                prediction.
        Returns:
            h_out: numpy.ndarray
                An `n` by `n_out` (# classes) matrix that corresponds to 
                the estimated class probabilities in the output layer. 
                Each row corresponds to the estimated probability for
                each data point in the current batch.
        """

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in
        
        # For each data point, this is `a_out = W.dot(h_in) + b`
        a_out = h_in.dot(self.W.T) + self.b.T

        if update_units:
            self.h_in = h_in

        # Softmax
        ex    = np.exp(a_out)
        h_out = ex / ex.sum(axis=1, keepdims=True)

        return h_out

    def bprop(self, grad_a_out):
        """
        Backpropagation of the gradient w.r.t. the *pre-softmax hidden units*.
        Updates model parameters (`W` and `b`) and returns the gradient w.r.t.
        the post-activation units in the downstream.

        For both the argument and the return value, each row corresponds to 
        the gradient from each data point in the current batch.

        Args: 
            grad_a_out: numpy.ndarray
                An `n` by `n_out` (# classes) matrix of gradients w.r.t.
                the *pre-softmax hidden units* in the immediate upstream. 
        Returns:
            grad_h_in: numpy.ndarray
                An `n` by `n_in` matrix of gradients with respect to
                the post-activation units in the immediate downstream.
        """

        # Assert that `fprop` is already done
        assert self.h_in is not None
        assert self.h_in.shape[0]  == grad_a_out.shape[0]
        n = self.h_in.shape[0]

        # Compute gradients
        grad_W     = (1. / n) * grad_a_out.T.dot(self.h_in)
        grad_decay = 2. * self.weight_decay * self.W
        grad_b     = grad_a_out.mean(axis=0, keepdims=True).T
        grad_h_in  = grad_a_out.dot(self.W)

        # SGD Updates
        lr = self.learning_rate.get()
        self.W = self.W - lr * (grad_W + grad_decay)
        self.b = self.b - lr * (grad_b)

        return grad_h_in
