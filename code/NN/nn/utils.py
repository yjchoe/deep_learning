"""
Utility functions
"""

from __future__ import division
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path_train, path_valid, path_test):
    """Loads data from three CSV files.

    Args:
        path_train, path_valid, path_test: string
            Relative paths to the training/validation/testing datasets.
    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test: numpy.ndarray
            Data matrices and outcome vectors for each of the datasets.
    Raises:
        IOError: An error occurred accessing one of the specified files.
    """

    data_train = np.genfromtxt(path_train, delimiter=',')
    data_valid = np.genfromtxt(path_valid, delimiter=',')
    data_test  = np.genfromtxt(path_test , delimiter=',')

    X_train, y_train = data_train[:, :-1], transform_y(data_train[:, -1].astype(int))
    X_valid, y_valid = data_valid[:, :-1], transform_y(data_valid[:, -1].astype(int))
    X_test , y_test  = data_test [:, :-1], transform_y(data_test [:, -1].astype(int))

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def transform_y(y, n_classes=None):
    """Transforms a categorical outcome vector to/from a one-hot vector.

    Args:
        y: numpy.ndarray
            Either
                An `n`-vector of categorical outcomes.
                Entries are one of `0`, `1`, ..., `c-1`.
            or
                An `n` by `c` binary matrix. 
                `y[i, k] == 1` iff `i`th data point belongs to class `k`.            
        n_classes: int
            Number of categorical outcomes. Defaults to `max(y)+1`.
    Returns:
        y: numpy.ndarray
            The other form of the input `y`.
    """
    if len(y.shape) == 1:
        if n_classes is None:
            n_classes = max(y) + 1
        return np.eye(len(y), n_classes, dtype=bool)[y, :]
    elif len(y.shape) == 2:
        return np.argmax(y, axis=1)
    else:
        raise Exception('utils.transform_y: ' +
                        'argument is not a proper outcome vector')

def binarize_data(X, threshold=0.5):
    """Binarize training inputs to 0 or 1.

    This is used for inputs to restricted Boltzmann machines.
    
    Args:
        X: numpy.ndarray
            Training data matrix scaled to be within [0, 1].

    Returns:
        X: numpy.ndarray (dtype: np.int8)
            Training data matrix thresholded at input `threshold`.
    """
    return (X >= threshold).astype(np.int8)

def standardize_data(X):
    """Standardize training inputs to zero mean and unit variance.
    
    A wrapper around `sklearn.preprocessing.StandardScaler()`.

    Args:
        X: numpy.ndarray
            Training data matrix

    Returns:
        scaler: sklearn.preprocessing.StandardScaler
            Scaler fitted to the training input data `X`.
            Use `scaler.fit(X_new)` to transform your validation/testing data.
            Use `scaler.mean_` and `scaler.scale_` for the original 
            mean and standard deviation.
    """
    return StandardScaler().fit(X)

def generate_batches(n, batch_size, batch_seed=None):
    """A generator for batches.

    Args:
        n: int
            Total number of data points to choose from.
        batch_size: int
            Number of data points per batch.
        batch_seed: int
            Random seed for ordering of data.

    Returns:
        A generator that each time yields a list of indices for a batch.
    """
    batch_size = min(batch_size, n)
    rng        = np.random.RandomState(batch_seed)
    perm       = rng.choice(np.arange(n), n, replace=False)
    for i in xrange(int(np.ceil(n / batch_size))):
        yield perm[i*batch_size:(i+1)*batch_size]
