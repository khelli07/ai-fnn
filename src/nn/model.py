import numpy as np
import copy
from .optimizer import SGD
from .loss import MeanSquaredError


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        if not self.layers and layer.input_dim is not None:
            self.is_built = True
        self.layers.append(layer)

    def _initialize_weights_and_bias(self, input_shape):
        for layer in self.layers:
            layer.initialize(input_shape, stdev=1)
            input_shape = layer.units

    def compile(self, optimizer=SGD(), loss=MeanSquaredError, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def evaluate(self, X, y):
        ypred = self.predict(X)
        loss = self.loss.forward(ypred, y)

        print("----- Summary -----")
        print(f"- {self.loss.name()} = {loss}")
        if self.metrics:
            metrics = self.metrics.calculate(ypred, y)
            print(f"- {self.metrics.name()} = {metrics}")
        print("-------------------")

    def predict(self, X):
        _, a = self.layers[0].forward(X.T)
        for j in range(1, len(self.layers)):
            _, a = self.layers[j].forward(a)
        return a

    def fit(
        self, X, y, batch_size=1, epochs=1, verbose=1, history=False, save_best=False
    ):

        if self.layers[0].input_dim is None:
            raise ValueError("Please specify input_dim on the first layer")

        total_samples = len(y)
        samples_per_batch = int(total_samples / batch_size)

        number_of_layers = len(self.layers)
        self._initialize_weights_and_bias(input_shape=self.layers[0].input_dim)

        losses = []
        tmp = list(zip(X, y))

        best_layers = None
        best_loss = np.inf

        for i in range(epochs):
            # Shuffle the data
            np.random.shuffle(tmp)
            X, y = zip(*tmp)
            X_taken, y_taken = (
                np.array(X)[:samples_per_batch].T,
                np.array(y)[:samples_per_batch],
            )

            # Forward pass
            z, a = self.layers[0].forward(X_taken)
            zs, activations = [z], [a]
            for j in range(1, number_of_layers):
                z, a = self.layers[j].forward(activations[-1])
                zs.append(z)
                activations.append(a)

            loss = self.loss.forward(activations[-1], y_taken)

            # -------------------------------
            if loss < best_loss:
                best_loss = loss
                best_layers = copy.deepcopy(self.layers)

            if history:
                losses.append(loss)

            if verbose == 1:
                print(f"Epoch ({i + 1}/{epochs}), loss = {loss}")
            # -------------------------------

            # Backward pass
            propagated_error = self.loss.backward(activations[-1][0], y_taken)
            for j in range(number_of_layers - 2, -1, -1):
                delta, propagated_error = self.layers[j].backward(
                    propagated_error, zs[j]
                )

                prev_activation = X_taken if j == 0 else activations[j - 1]

                nabla_w, nabla_b = self.optimizer.generate_nabla(delta, prev_activation)
                self.layers[j].update(nabla_w, nabla_b)

        print(f"Epoch ({i + 1}/{epochs}), loss = {loss}")

        if save_best:
            self.layers = best_layers
            print("Best loss:", best_loss)

        return losses
