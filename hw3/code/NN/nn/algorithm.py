"""
Training algorithms and related classes
"""

import numpy as np

class LearningRate(object):
    """Learning rate function class.

    Attributes:
        const: float
            Constant term for the learning rate.
        epoch: int
            Current epoch number, updated from `NN.train`.
    """
    def __init__(self, const, epoch=0):
        self.const = const
        self.epoch = epoch

    def lr(self, t):
        return self.const / (t // 100 + 1.)

    def get(self):
        return self.lr(self.epoch)
