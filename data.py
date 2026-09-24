import numpy as np


# Create and return binary inputs and decimal targets.
def get_data():
    # Each row represents [x1, x2, x3].
    X = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])

    # The target is the decimal value represented by each binary input.
    y = np.array([1, 2, 3, 4, 5, 6, 7])

    return X, y