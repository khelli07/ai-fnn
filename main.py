import numpy as np
from src.nn.model import Sequential
from src.nn.layer import Dense
from src.nn.activation import RelU
from src.nn.optimizer import SGD

X = np.array([[1, 5], [2, 8], [3, 2], [4, 6]])
y = np.array([14, 25, 9, 23])

np.random.seed(0)

model = Sequential()
model.add(Dense(3, activation=RelU, input_dim=2))
model.add(Dense(2, activation=RelU))
model.add(Dense(1, activation=RelU))

model.compile(optimizer=SGD(learning_rate=1e-4))

model.fit(X, y, epochs=50000)

print(model.predict(X.T))

# Epoch (50000/50000), loss = 0.0006393971345776127
# [[14.02944217 24.98119096  8.99962981 22.99912095]]
