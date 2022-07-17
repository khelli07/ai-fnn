import numpy as np
from utils import *

x = np.array([[1, 2, 3, 4], [5, 8, 2, 6]])
y = np.array([14, 25, 9, 23])  # 2x1 + 3x2 - 3
# y = np.array([2, 9, 8, 19])  # x1^2 + x2 - 3
number_of_samples = len(x[0])

np.random.seed(0)

z = []
a = []
w = []
b = []

neurons = [2, 3, 2, 1]
number_of_layers = len(neurons) - 1
lr = 1e-4  # 0.0001 = 10^-4
epochs = 10000

# Forward propagation
"""
    Architecture:
        2 neurons (input),
        3 neurons (hidden),
        2 neurons (hidden),
        1 neurons (output)
"""

# FIRST HIDDEN LAYER
w.append(np.random.randn(neurons[1], neurons[0]))
b.append(np.ones((neurons[1], 1)))
z.append(w[0] @ x + b[0])
a.append(RelU_forward(z[0]))

assert a[0].shape == (neurons[1], number_of_samples)

# SECOND HIDDEN LAYER
w.append(np.random.randn(neurons[2], neurons[1]))
b.append(np.ones((neurons[2], 1)))
z.append(w[1] @ a[0] + b[1])
a.append(RelU_forward(z[1]))

assert a[1].shape == (neurons[2], number_of_samples)

# OUTPUT LAYER
w.append(np.random.randn(neurons[3], neurons[2]))
b.append(np.ones((neurons[3], 1)))
z.append(w[2] @ a[1] + b[2])
a.append(RelU_forward(z[2]))

assert a[2].shape == (neurons[3], number_of_samples)

print(f"Epoch (1/{epochs}), loss = {cost_forward(a[2], y)}")

# Backward propagation
# OUTPUT LAYER
delta = cost_backward(a[2][0], y) * RelU_backward(z[2])
w[2] = w[2] - lr * (delta @ a[1].T)
b[2] = b[2] - lr * (delta)

# SECOND HIDDEN LAYER
delta = (w[2].T @ delta) * RelU_backward(z[1])
w[1] = w[1] - lr * (delta @ a[0].T)
b[1] = b[1] - lr * (delta)

#  FIRST HIDDEN LAYER
delta = (w[1].T @ delta) * RelU_backward(z[0])
w[0] = w[0] - lr * (delta @ x.T)
b[0] = b[0] - lr * (delta)

for i in range(1, epochs):
    # Forward propagation
    for j in range(number_of_layers):
        if j == 0:
            z[j] = w[j] @ x + b[j]
        else:
            z[j] = w[j] @ a[j - 1] + b[j]

        a[j] = RelU_forward(z[j])

    print(f"Epoch ({i + 1}/{epochs}), loss = {cost_forward(a[2], y)}")

    # Backward propagation
    for j in range(number_of_layers - 1, -1, -1):
        if j == number_of_layers - 1:
            delta = cost_backward(a[j][0], y) * RelU_backward(z[j])
        else:
            delta = (w[j + 1].T @ delta) * RelU_backward(z[j])

        if j == 0:
            w[j] = w[j] - lr * (delta @ x.T)
        else:
            w[j] = w[j] - lr * (delta @ a[j - 1].T)

        b[j] = b[j] - lr * (delta)

print("last y_train predicted (a[2]) =", a[2])

# Epoch (10000/10000), loss = 0.00030291977689243294
# last y_train predicted (a[2]) = [[14.02074786 24.98681486  8.99915625 23.001178  ]]

"""
    Limitations:
    Be careful in choosing
        - learning rate,
        - weight and bias initialization technique,
        - and epochs
    It should be tweaked to achieve the desired result.
    Otherwise, an exploding gradient will occur.
"""
