"""
Neural network layer class
"""

import numpy as np

class Layer(object):
    """A base class for layers of a feedforward neural network.

    Includes the weight matrix and the bias vector as well as 
    the hyperparameters for training. In the current implementation,
    this class is always instantiated by the `NN.__init__` method.

    Attributes (comes from `NN.__init__`):
        __name__: string
            Name of the layer such as `layer3`.
        n_in: int
            Number of input units to the layer.
        n_out: int
            Number of output units to the layer.
        learning_rate: function(epoch) or float
            A function that takes the current epoch number and returns 
            a learning rate. 
        momentum: float (between 0 and 1)
            Momentum parameter for exponential averaging of previous 
            gradients.
        weight_decay: float
            A weight decay / L2 regularization parameter.
        dropout: float (between 0 and 1)
            Probability of a unit getting dropped out. 
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
                 learning_rate, momentum, weight_decay, dropout, seed): 
        """
        Neural network layer initializer.
        """

        # Attributes
        self.__name__      = name
        self.n_in          = n_in
        self.n_out         = n_out
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.dropout       = dropout
        self.seed          = seed

        # Weight and bias initialization
        self.rng = np.random.RandomState(seed)
        c = np.sqrt(6. / (n_in + n_out))
        self.W = self.rng.uniform(-c, c, size=(n_out, n_in))
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
                 learning_rate, momentum, weight_decay, dropout, seed): 
        """
        Neural network hidden layer initializer.
        """

        # Initialization from the superclass
        super(HiddenLayer, self).__init__(
            name, n_in, n_out, learning_rate, momentum, 
            weight_decay, dropout, seed)

        # Gradients (initially zero; set to matrices for momentum calculation)
        self.grad_a_out = np.zeros((2, self.n_out))  # involves column-wise mean
        self.grad_W     = 0.0
        self.grad_decay = 0.0
        self.grad_b     = 0.0
        self.grad_h_in  = np.zeros((2, self.n_in))   # involves column-wise mean

        # Additional attributes
        self.activation = activation
        self.h_in       = None
        self.a_out      = None

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

        h_out = self.activation.eval(a_out)

        # Dropout
        if update_units:
            mask = self.rng.binomial(1, 1.-self.dropout, size=h_out.shape)
        else:
            # Taking expectation at test time
            mask = (1.-self.dropout) * np.ones(h_out.shape)  
        return h_out * mask

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
        self.grad_a_out = grad_h_out * self.activation.grad(self.a_out) \
                            + self.momentum * self.grad_a_out.mean(axis=0)
        self.grad_W     = (1. / n) * self.grad_a_out.T.dot(self.h_in) \
                            + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W \
                            + self.momentum * self.grad_decay
        self.grad_b     = self.grad_a_out.mean(axis=0, keepdims=True).T \
                            + self.momentum * self.grad_b
        self.grad_h_in  = self.grad_a_out.dot(self.W) \
                            + self.momentum * self.grad_h_in.mean(axis=0)

        # SGD Updates
        lr = self.learning_rate.get()
        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * (self.grad_b)

        return self.grad_h_in

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
                 learning_rate, momentum, weight_decay, dropout, seed): 
        """
        Neural network output layer initializer.
        """

        # Initialization from the superclass
        super(OutputLayer, self).__init__(
            name, n_in, n_out, learning_rate, momentum, 
            weight_decay, dropout, seed)

        # Gradients (initially zero; set to matrices for momentum calculation)
        self.grad_W     = 0.0
        self.grad_decay = 0.0
        self.grad_b     = 0.0
        self.grad_h_in  = np.zeros((2, self.n_in))  # involves column-wise mean

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
        h_out = ex / (ex.sum(axis=1, keepdims=True) + 1e-8)

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

        # Compute gradients and store them for next iteration (momentum)
        self.grad_W     = (1. / n) * grad_a_out.T.dot(self.h_in) \
                            + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W \
                            + self.momentum * self.grad_decay
        self.grad_b     = grad_a_out.mean(axis=0, keepdims=True).T \
                            + self.momentum * self.grad_b
        self.grad_h_in  = grad_a_out.dot(self.W) \
                            + self.momentum * self.grad_h_in.mean(axis=0)

        # SGD Updates
        lr = self.learning_rate.get()
        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * (self.grad_b)

        return self.grad_h_in

class AEInputLayer(HiddenLayer):
    """A subclass of `HiddenLayer` for autoencoder input layer.

    Exactly same functionalities as `HiddenLayer`, except that gradients of
    the tied weights are output by `bprop`.

    """
    def __init__(self, name, n_in, n_out, activation, 
                 learning_rate, momentum, weight_decay, dropout, seed): 
        """
        Autoencoder input layer initializer.
        """

        # Initialization from the superclass
        super(AEInputLayer, self).__init__(
            name, n_in, n_out, activation, 
            learning_rate, momentum, weight_decay, dropout, seed)

    def bprop(self, grad_h_out):
        """
        Only difference from the HiddenLayer version is that gradient updates
        for the tied weight matrix is deferred to the main training loop.

        Args: 
            grad_h_out: numpy.ndarray
                An `n` by `n_hidden` matrix of gradients w.r.t.
                the *post-activation* hidden units in the immediate upstream. 
        Returns:
            grad_h_in: numpy.ndarray
                An `n` by `n_visible` matrix of gradients with respect to
                the post-activation units in the immediate downstream.
        """

        # Assert that `fprop` is already done
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0]  == grad_h_out.shape[0]
        assert self.a_out.shape[0] == grad_h_out.shape[0]
        n = self.h_in.shape[0]

        # Compute gradients
        self.grad_a_out = grad_h_out * self.activation.grad(self.a_out) \
                            + self.momentum * self.grad_a_out.mean(axis=0)
        self.grad_W     = (1. / n) * self.grad_a_out.T.dot(self.h_in) \
                            + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W \
                            + self.momentum * self.grad_decay
        self.grad_b     = self.grad_a_out.mean(axis=0, keepdims=True).T \
                            + self.momentum * self.grad_b
        self.grad_h_in  = self.grad_a_out.dot(self.W) \
                            + self.momentum * self.grad_h_in.mean(axis=0)

        # SGD Updates
        lr = self.learning_rate.get()
#        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * (self.grad_b)

        return self.grad_h_in



class AEOutputLayer(Layer):
    """A subclass of `Layer` for the autoencoder output layer.
    Analogous to `HiddenLayer` but accepts pre-activation gradients during bprop.
    See `.autoencoder.Autoencoder` class for details.

    Additional inputs:
        W: numpy.ndarray
            Tied weight matrix from the forward `HiddenLayer` of the autoencoder.
            Size `n_hidden` by `n_visible`.

    Additional attributes:
        activation: Activation
            An instance of the Activation class.

    Additional non-input attributes:
        h_in, a_out: numpy.ndarray
            Intermediate unit values during backpropagation.

    Methods:
        __init__, fprop, bprop
    """

    def __init__(self, name, n_in, n_out, W, activation, 
                 learning_rate, momentum, weight_decay, dropout, seed): 
        """
        Autoencoder output layer initializer.
        """

        # Initialization from the superclass
        # n_in (n_hidden) and n_out (n_visible) are redundant
        super(AEOutputLayer, self).__init__(
            name, n_in, n_out, learning_rate, momentum, 
            weight_decay, dropout, seed)

        # Tied weights: `self.W.T` is the fprop weights. 
        # Note that bias is not shared and is initialized at random above.
        self.W = W

        # Gradients (initially zero; set to matrices for momentum calculation)
        self.grad_a_out = np.zeros((2, self.n_out))  # involves column-wise mean
        self.grad_W     = 0.0
        self.grad_decay = 0.0
        self.grad_b     = 0.0
        self.grad_h_in  = np.zeros((2, self.n_in))   # involves column-wise mean

        # Additional attributes
        self.activation = activation
        self.h_in       = None
        self.a_out      = None

    def fprop(self, h_in, update_units=False):
        """
        Forward propagation of incoming units through the current layer.
        Includes a linear transformation and an activation. 
        Can accept batch inputs of size `n`.

        Args:
            h_in: numpy.ndarray
                An `n` by `n_hidden` matrix that corresponds to 
                the hidden units from the immediate downstream. 
                Each row corresponds to the hidden unit value from
                each data point in the current batch.
            update_units: boolean
                If `True`, stores `h_in` and `a_in` that are later used for 
                backpropagation. The default `False` option is used for 
                prediction.
        Returns:
            h_out: numpy.ndarray
                An `n` by `n_visible` matrix that corresponds to 
                the activated hidden units in the immediate upstream. 
                Each row corresponds to the hidden unit value from
                each data point in the current batch.
        """

        assert isinstance(h_in, np.ndarray)
        assert h_in.shape[1] == self.n_in
        
        # For each data point, this is `a_out = W.T.dot(h_in) + b`
        a_out = h_in.dot(self.W) + self.b.T

        if update_units:
            self.h_in  = h_in
            self.a_out = a_out

        h_out = self.activation.eval(a_out)

        # Dropout
        if update_units:
            mask = self.rng.binomial(1, 1.-self.dropout, size=h_out.shape)
        else:
            # Taking expectation at test time
            mask = (1.-self.dropout) * np.ones(h_out.shape)  
        return h_out * mask

    def bprop(self, grad_a_out):
        """
        Backpropagation of the gradient w.r.t. the *pre-activation* units 
        in the upstream (output direction).
        Updates model parameters (`W` and `b`) and returns the gradient w.r.t.
        the post-activation units in the downstream.

        For both the argument and the return value, each row corresponds to 
        the gradient from each data point in the current batch.

        Args: 
            grad_a_out: numpy.ndarray
                An `n` by `n_visible` matrix of gradients with respect to
                the *pre-activation* units in the immediate upstream. 
        Returns:
            grad_h_in: numpy.ndarray
                An `n` by `n_hidden` matrix of gradients with respect to
                the post-activation units in the immediate downstream.
        """

        # Assert that `fprop` is already done
        assert self.h_in is not None and self.a_out is not None
        assert self.h_in.shape[0]  == grad_a_out.shape[0]
#        assert self.a_out.shape[0] == grad_a_out.shape[0]
        n = self.h_in.shape[0]

        # Compute gradients
        # Tied weight gradients use the transpose.
        self.grad_W     = (1. / n) * self.h_in.T.dot(grad_a_out) \
                            + self.momentum * self.grad_W
        self.grad_decay = 2. * self.weight_decay * self.W \
                            + self.momentum * self.grad_decay
        self.grad_b     = grad_a_out.mean(axis=0, keepdims=True).T \
                            + self.momentum * self.grad_b
        self.grad_h_in  = grad_a_out.dot(self.W.T) \
                            + self.momentum * self.grad_h_in.mean(axis=0)

        # SGD updates
        # The actual update happens in the `Autoencoder.train` loop.
        lr = self.learning_rate.get()
#        self.W = self.W - lr * (self.grad_W + self.grad_decay)
        self.b = self.b - lr * (self.grad_b)

        return self.grad_h_in
