"""
Utility functions
"""

import numpy as np

def load_data(path_train, path_valid, path_test):
    """
    Loads data from three CSV files.

    Args:
        path_train, path_valid, path_test: string
            Relative paths to the training/validation/testing datasets.
    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test: numpy.ndarray
            Data matrices and target vectors for each of the datasets.
    Raises:
        IOError: An error occurred accessing one of the specified files.
    """

    data_train = np.genfromtxt(path_train, delimiter=',')
    data_valid = np.genfromtxt(path_valid, delimiter=',')
    data_test  = np.genfromtxt(path_test , delimiter=',')

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_valid, y_valid = data_valid[:, :-1], data_valid[:, -1]
    X_test , y_test  = data_test [:, :-1], data_test [:, -1]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
