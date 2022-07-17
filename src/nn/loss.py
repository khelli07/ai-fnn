import numpy as np


class MeanSquaredError:
    def forward(ypred, ytrue):
        return 1 / 2 * (np.sum(np.square(ytrue - ypred)))

    def backward(ypred, ytrue):
        return ypred - ytrue


class BinaryCrossentropy:
    def forward(ypred, ytrue):
        return

    def backward(ypred, ytrue):
        return
