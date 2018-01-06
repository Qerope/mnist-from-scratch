import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(float)


def softmax(z):
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


class Network:
    def __init__(self, sizes, hidden='sigmoid', output='softmax'):
        self.sizes = sizes
        self.f, self.fp = {'sigmoid': (sigmoid, sigmoid_prime),
                           'relu': (relu, relu_prime)}[hidden]
        self.out = {'softmax': softmax, 'sigmoid': sigmoid}[output]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            a = self.out(z) if i == len(self.weights) - 1 else self.f(z)
        return a

    def backprop(self, X, Y):
        """X is (n_in, m), Y is (n_out, m); returns gradients summed over the batch."""
        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        a = X
        activations = [X]
        zs = []
        last = len(self.weights) - 1
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.out(z) if i == last else self.f(z)
            activations.append(a)
        delta = activations[-1] - Y
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, len(self.sizes)):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.fp(zs[-l])
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
        history = []
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
                history.append(acc)
                print('epoch %d: %.4f' % (epoch + 1, acc))
        return history

    def save(self, path):
        np.savez(path, *self.weights, *self.biases)

    def load(self, path):
        data = np.load(path)
        k = len(self.weights)
        self.weights = [data['arr_%d' % i] for i in range(k)]
        self.biases = [data['arr_%d' % i] for i in range(k, 2 * k)]
