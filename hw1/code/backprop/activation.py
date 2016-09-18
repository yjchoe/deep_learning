"""
Activation functions
"""

import numpy as np
from scipy.special import expit

class Activation(object):
    """A class of activation functions for neural networks.

    Attributes:
        type: string
            Either 'sigmoid' (default), 'tanh', 'relu', or 'linear'.
    Methods:
        eval, grad
    """

    def __init__(self, type='sigmoid'):
        self.type = type;
        if self.type not in ['sigmoid', 'tanh', 'relu', 'linear']:
            raise Exception('Activation.__init__: ' + 
                            'Activation type not recognized')

    def eval(self, a):
        """
        Evaluate the activation at value `a` (vectorized). 
        """
        if self.type == 'sigmoid':
            return expit(a)
        elif self.type == 'tanh':
            return np.tanh(a)
        elif self.type == 'relu':
            return a * (a > 0)
        else:  # linear
            return a

    def grad(self, a):
        """
        Compute the gradient of the activation at value `a` (vectorized).
        """
        if self.type == 'sigmoid':
            s = self.eval(a)
            return s * (1. - s)
        elif self.type == 'tanh':
            return 1. - self.eval(a) ** 2
        elif self.type == 'relu':
            return a > 0
        else:  # linear
            return 1.
