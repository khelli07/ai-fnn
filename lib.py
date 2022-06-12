import numpy as np


# Activation
def RelU_forward(inputs):
    return np.maximum(0, inputs, dtype=np.float64)


def RelU_backward(inputs):
    return np.array(inputs > 0, dtype=np.float64)


# Cost
def cost_forward(ypred, ytrue):
    n = len(ytrue)
    return (1 / (2 * n)) * (np.sum(np.square(ytrue - ypred)))


def cost_backward(ypred, ytrue):
    return ypred - ytrue
