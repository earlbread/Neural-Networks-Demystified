import numpy as np

class NeuralNetwork():
    def __init__(self):
        # Hyper parameters
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y_hat = self.sigmoid(self.z3)

        return y_hat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, y, y_hat):
        diff = y - y_hat
        return np.mean(np.sum(diff ** 2))
