import numpy as np
from src.nn.model import Sequential
from src.nn.layer import Dense
from src.nn.activation import RelU, Sigmoid
from src.nn.optimizer import SGD
from src.nn.metrics import Accuracy

X = np.array([[1, 5], [2, 8], [3, 2], [4, 6]])
y = np.array([0, 1, 0, 1])
# y = np.array([14, 25, 9, 23])

np.random.seed(0)

model = Sequential()
model.add(Dense(32, activation=RelU, input_dim=2))
model.add(Dense(32, activation=RelU))
model.add(Dense(1, activation=Sigmoid))

model.compile(optimizer=SGD(learning_rate=5e-4))

model.fit(X, y, epochs=250)

model.evaluate(X, y)

ypred = np.array(model.predict(np.array(X)) > 0.5, dtype=np.int32)
print(Accuracy.calculate(ypred, y))  # 1.0
