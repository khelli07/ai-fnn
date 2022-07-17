import numpy as np
from .optimizer import SGD
from .loss import MeanSquaredError


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
        self.is_built = False

    def add(self, layer):
        if not self.layers and layer.input_dim is not None:
            self.is_built = True
        self.layers.append(layer)

    def _initialize_weights_and_bias(self, input_shape):
        # stdev = 1 / np.sqrt(input_shape)
        for layer in self.layers:
            layer.initialize(input_shape, stdev=1)
            input_shape = layer.units

    def compile(self, optimizer=SGD(), loss=MeanSquaredError, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def evaluate(self, X, y):
        return

    def predict(self, X):
        _, a = self.layers[0].forward(X)
        for j in range(1, len(self.layers)):
            _, a = self.layers[j].forward(a)
        return a

    def fit(self, X, y, batch_size=1, epochs=1, validation_data=None):
        total_samples = len(y)
        samples_per_batch = int(total_samples / batch_size)

        number_of_layers = len(self.layers)
        self._initialize_weights_and_bias(input_shape=self.layers[0].input_dim)

        for i in range(epochs):
            tmp = list(zip(X, y))
            np.random.shuffle(tmp)
            X, y = zip(*tmp)
            X, y = np.array(X), np.array(y)
            X_taken = X[:samples_per_batch]
            y_taken = y[:samples_per_batch]

            z, a = self.layers[0].forward(X_taken.T)
            zs, activations = [z], [a]
            for j in range(1, number_of_layers):
                z, a = self.layers[j].forward(activations[-1])
                zs.append(z)
                activations.append(a)

            loss = self.loss.forward(activations[-1], y_taken)
            print(f"Epoch ({i + 1}/{epochs}), loss = {loss}")

            propagated_error = self.loss.backward(activations[-1][0], y_taken)
            for j in range(number_of_layers - 1, -1, -1):
                delta, propagated_error = self.layers[j].backward(
                    propagated_error, zs[j]
                )
                if j == 0:
                    prev_layer_output = X_taken.T
                else:
                    prev_layer_output = activations[j - 1]

                nabla_w, nabla_b = self.optimizer.generate_nabla(
                    delta, prev_layer_output
                )
                self.layers[j].optimize(nabla_w, nabla_b)
