import numpy as np
from lib import *

x = np.array([[1, 2, 3, 4], [5, 8, 2, 6]])
y = np.array([14, 25, 9, 23])  # 2x1 + 3x2 - 3
number_of_samples = len(x[0])

np.random.seed(0)

z, a, w, b = [], [], [], []

neurons = [2, 3, 2, 1]
number_of_layers = len(neurons) - 1
lr = 1e-4  # 0.0001 = 10^-4
epochs = 10000

# -- INITIALIZATION
a.append(x)
for j in range(number_of_layers):
    w.append(np.random.randn(neurons[j + 1], neurons[j]))
    b.append(np.ones((neurons[j + 1], 1)))
    z.append(w[j] @ a[j] + b[j])
    a.append(RelU_forward(z[j]))

# -- LEARNING
for i in range(epochs):
    # Forward propagation
    for j in range(number_of_layers):
        z[j] = w[j] @ a[j] + b[j]
        a[j + 1] = RelU_forward(z[j])

    print(f"Epoch ({i + 1}/{epochs}), loss = {cost_forward(a[3], y)}")

    # Backward propagation
    delta = cost_backward(a[number_of_layers][0], y)
    for j in range(number_of_layers - 1, -1, -1):
        if j != number_of_layers - 1:
            delta = w[j + 1].T @ delta

        delta = delta * RelU_backward(z[j])
        w[j] = w[j] - lr * (delta @ a[j].T)
        b[j] = b[j] - lr * (delta)
