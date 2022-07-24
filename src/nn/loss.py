import numpy as np


class MeanSquaredError:
    @staticmethod
    def forward(ypred, ytrue):
        return 1 / 2 * (np.sum(np.square(ytrue - ypred)))

    @staticmethod
    def backward(ypred, ytrue):
        return ypred - ytrue

    def name():
        return "mean_squared_error"


class BinaryCrossentropy:
    @staticmethod
    def forward(ypred, ytrue):
        n = len(ytrue)
        return (-1 / n) * np.sum(
            np.nan_to_num(-ytrue * np.log(ypred) - (1 - ytrue) * np.log(1 - ypred))
        )

    @staticmethod
    def backward(ypred, ytrue):
        return np.nan_to_num((-ytrue / ypred) + ((1 - ytrue) / (1 - ypred)))

    def name():
        return "binary_crossentropy"
