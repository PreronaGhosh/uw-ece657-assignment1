import numpy as np
import pandas as pd

x = pd.read_csv('C:/Users/prero/OneDrive/Desktop/train_data.csv', header=None)
y = pd.read_csv("C:/Users/prero/OneDrive/Desktop/train_labels.csv", header=None)
data = np.append(x, y, axis=1)
print(data)
print("Dataset dimension: ", data.shape)

np.random.shuffle(data)
r, c = data.shape
split = 0.2
train_set = data[int(r * split):]  # 80% of the whole dataset
test_set = data[:int(r * split)]  # 20% of the whole dataset

train_x = train_set[:, :-4]
train_y = train_set[:, -4:]

test_x = test_set[:, :-4]
test_y = test_set[:, -4:]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights1 = 0.1 * np.random.randn(self.input_dim, self.hidden_dim)
        self.bias1 = np.zeros((1, self.hidden_dim))

        self.weights2 = 0.1 * np.random.randn(self.hidden_dim, self.output_dim)
        self.bias2 = np.zeros((1, self.output_dim))

    def forward(self, X):
        self.hidden_layer = np.dot(X, self.weights1) + self.bias1
        self.hidden_activation = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_activation, self.weights2) + self.bias2
        self.output_activation = self.softmax(self.output_layer)

        return self.output_activation

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Calculate gradients
        output_error = self.output_activation - y
        hidden_error = np.dot(output_error, self.weights2.T) * self.sigmoid_derivative(self.hidden_layer)

        weights2_gradient = np.dot(self.hidden_activation.T, output_error) / m
        bias2_gradient = np.sum(output_error) / m

        weights1_gradient = np.dot(X.T, hidden_error) / m
        bias1_gradient = np.sum(hidden_error) / m

        # Update weights and biases
        self.weights2 -= learning_rate * weights2_gradient
        self.bias2 -= learning_rate * bias2_gradient

        self.weights1 -= learning_rate * weights1_gradient
        self.bias1 -= learning_rate * bias1_gradient

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            loss = self.cross_entropy_loss(y, output)
            print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss


# Code to train and test the above neural network
NN = MLP(784, 64, 4)
NN.train(train_x, train_y, 100, 0.01)
NN.predict(train_x)
print("Output values: ")
print(NN.predict(test_x))
print("Actual test_y: ", test_y)