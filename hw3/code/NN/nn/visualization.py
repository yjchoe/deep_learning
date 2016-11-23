"""
Visualization tools (MNIST)
"""

import numpy as np
import matplotlib.pyplot as plt


def print_image(X, output_shape=None, title=''):
    """Prints a set of 2D images of size d * d.

    Args:
        X: numpy.ndarray
            A data matrix of size `n` by `d**2`, where each row corresponds to
            a grayscale image of size `d` by `d`, that is flatten by a 
            row-major ordering.
        output_shape: tuple of ints (m=n_row, k=n_col)
            An optional argument specifying in what shape the `n` images are
            printed out. If `None`, images will be printed out horizontally. 
            If `(m, k)`, where `n == m * k`, the `n` images will be printed in
            a row-major ordering.
    Returns:
        None
    """

    n, dsq = X.shape
    if n > 400:
        raise Exception('print_image: too many input images (more than 400)')
    d = np.sqrt(dsq)
    if np.round(d) == d:
        d = int(d)
    else:
        raise Exception('print_image: input image is not square')
    if output_shape is not None:
        m, k = output_shape
        assert n == m * k
    else:
        m, k = n, 1

    fig = plt.figure(figsize=(k, m)) 
    plt.gray()
    for i in range(m*k):
        ax = fig.add_subplot(m, k, i+1)
        ax.matshow(X[i, :].reshape(d, d)) # row-major
        ax.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)
    return fig
