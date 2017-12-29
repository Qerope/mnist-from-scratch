import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, X, Y):
        """X is (n_in, m), Y is (n_out, m); returns gradients summed over the batch."""
        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        a = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        delta = activations[-1] - Y
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, len(self.sizes)):
            delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def cost(self, X, Y):
        a = self.feedforward(X.T)
        return -np.sum(Y.T * np.log(a + 1e-12)) / len(X)

    def evaluate(self, X, labels):
        pred = np.argmax(self.feedforward(X.T), axis=0)
        return np.mean(pred == labels)

    def sgd(self, X, Y, epochs, batch_size, eta, lmbda=0.0, test=None):
        n = len(X)
        perm = np.random.permutation(n)
        for epoch in range(epochs):
            for k in range(0, n, batch_size):
                idx = perm[k:k + batch_size]
                nb, nw = self.backprop(X[idx].T, Y[idx].T)
                decay = 1 - eta * lmbda / n
                self.weights = [decay * w - eta / len(idx) * g for w, g in zip(self.weights, nw)]
                self.biases = [b - eta / len(idx) * g for b, g in zip(self.biases, nb)]
            if test is not None:
                acc = self.evaluate(*test)
                print('epoch %d: %.4f' % (epoch + 1, acc))
