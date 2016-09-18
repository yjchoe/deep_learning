"""
Training algorithms and related classes
"""

import numpy as np

class LearningRate(object):
    """
    TODO: Learning rate function class.

    Attributes:
        const: float
            Constant term for the learning rate.
        momentum: float (between 0 and 1)
            Momentum constant.
        epoch: int
            Current epoch, which is the number of times that `NN.train`
            was called.
    """
    def __init__(self, lr):
        self.lr = lr

    def get(self):
        return self.lr
