import numpy as np


class RelU:
    def forward(inputs):
        return np.maximum(0, inputs, dtype=np.float64)

    def backward(inputs):
        return np.array(inputs > 0, dtype=np.float64)


class Sigmoid:
    def forward(inputs):
        return 1.0 / (1.0 + np.exp(-inputs, dtype=np.float64))

    def backward(inputs):
        tmp = Sigmoid.forward(inputs)
        return tmp * (1.0 - tmp)
