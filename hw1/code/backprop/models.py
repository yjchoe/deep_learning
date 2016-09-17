"""
Feedforward Neural Network Models
"""

class NN(object):
    """A simple neural network class.

    Fully-connected feedforward single- or multi-layer neural networks 
    with nonlinear activation functions, weight decay and dropouts. 

    Attributes:
        architecture: tuple of ints
            A tuple of integers that contain the number of neurons per layer.
            For example, `(784, 100, 100, 10)` indicates a two-hidden-layer NN
            with 784-dimensional inputs, 100 hidden neurons in each of the 
            two hidden layers, and a 10-dimensional output.

        activation: string
            Choice of nonlinear functions. 
            Either 'sigmoid' (default), 'tanh', or 'relu'. 

        learning_rate: function(epoch) or float
            A function that takes the current epoch number and returns 
            a learning rate. Defaults to `momentum(eps=0.01, beta=0.5)`.
            If a float `b` is provided, defaults to `lambda epoch: b`.

        weight_decay: float
            A weight decay / L2 regularization parameter. Defaults to `1`.

        dropout: float (between 0 and 1)
            Probability of a unit getting dropped out. Defaults to `0.5`.

        early_stopping: boolean
            If true (default), attempts to stop training 
            before the validation error starts to increase.

    Methods:
        __init__, train, compute_error, predict

    """


    