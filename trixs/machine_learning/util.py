
import numpy as np


def onehot_reverse(y):
    """Takes a one-hot encoded list `y` and returns a numpy array of dimension
    (m, c) (see oneshot_forward) compatible with a Keras ML algorithm."""

    m = len(y)
    c = max(y) + 1

    yf = np.zeros((m, c))

    for i, yy in enumerate(y):
      yf[i, yy] = 1

    return yf
