import numpy as np


class RelU:
    @staticmethod
    def forward(inputs):
        return np.maximum(0, inputs, dtype=np.float64)

    @staticmethod
    def backward(inputs):
        return np.array(inputs > 0, dtype=np.float64)


class Sigmoid:
    @staticmethod
    def forward(inputs):
        logmax = 709.78
        inputs[-inputs > logmax] = logmax
        return 1.0 / (1.0 + np.exp(-inputs, dtype=np.float64))

    @staticmethod
    def backward(inputs):
        tmp = Sigmoid.forward(inputs)
        return tmp * (1.0 - tmp)
