import numpy as np
from neural_network import NeuralNetwork

X = np.array([[3, 5], [5, 1], [10, 2]])
y = np.array([0.75, 0.82, 0.93])

nn = NeuralNetwork()

y_hat = nn.forward(X)

print(X)
print(y)
print(y_hat)
print(nn.cost(y, y_hat))
