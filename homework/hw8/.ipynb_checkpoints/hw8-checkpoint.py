import numpy as np
import pandas as pd
import jax
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

datapath = "./"


def train_test_split(*arrays, test_size=0.2, shuffle=True, rand_seed=0):
    # set the random state if provided
    np.random.seed(rand_seed)

    # initialize the split index
    array_len = len(arrays[0].T)
    split_idx = int(array_len * (1 - test_size))

    # initialize indices to the default order
    indices = np.arange(array_len)

    # shuffle the arrays if shuffle is True
    if shuffle:
        np.random.shuffle(indices)

    # Split the arrays
    result = []
    for array in arrays:
        if shuffle:
            array = array[:, indices]
        train = array[:, :split_idx]
        test = array[:, split_idx:]
        result.extend([train, test])

    return result


def run_task1():
    csvname = datapath + 'new_gene_data.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:-1, :]
    y = data[-1:, :]

    print(np.shape(x))  # (7128, 72)
    print(np.shape(y))  # (1, 72)

    np.random.seed(0)  # fix randomness
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, rand_seed=0)

    # TODO


if __name__ == '__main__':
    run_task1()
