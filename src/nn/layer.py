import numpy as np


class Dense:
    def __init__(self, units: int, activation=None, input_dim: int = False):
        self.units = units
        self.activation = activation
        self.input_dim = input_dim

        self.weights = None
        self.bias = None

    def initialize(self, prev_units: int, mean: float = 0.0, stdev: float = 1.0):
        self.weights = np.random.normal(
            loc=mean, scale=stdev, size=(self.units, prev_units)
        )
        self.bias = np.ones((self.units, 1))

    def forward(self, inputs):
        z = self.weights @ inputs + self.bias
        if self.activation is not None:
            return z, self.activation.forward(z)
        return z, z

    def backward(self, propagated_error, current_z):
        delta = propagated_error
        if self.activation is not None:
            delta = delta * self.activation.backward(current_z)
        next_propagated_error = self.weights.T @ delta
        return delta, next_propagated_error

    def update(self, nabla_w, nabla_b):
        self.weights = self.weights - nabla_w
        self.bias = self.bias - nabla_b
        self.bias = np.average(self.bias, axis=1).reshape(self.units, 1)
