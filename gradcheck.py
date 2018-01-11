import numpy as np

from network import Network


def numeric_grad(net, X, Y, param, eps=1e-5):
    g = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        up = net.cost(X, Y)
        param[i] = old - eps
        down = net.cost(X, Y)
        param[i] = old
        g[i] = (up - down) / (2 * eps)
        it.iternext()
    return g


if __name__ == '__main__':
    np.random.seed(0)
    net = Network([6, 5, 3], hidden='sigmoid', output='softmax')
    X = np.random.randn(4, 6)
    Y = np.eye(3)[np.random.randint(0, 3, 4)]
    nb, nw = net.backprop(X.T, Y.T)
    for l, (w, gw) in enumerate(zip(net.weights, nw)):
        num = numeric_grad(net, X, Y, w) * len(X)
        print('layer %d weights: max abs error %.2e' % (l, np.abs(num - gw).max()))
    for l, (b, gb) in enumerate(zip(net.biases, nb)):
        num = numeric_grad(net, X, Y, b) * len(X)
        print('layer %d biases:  max abs error %.2e' % (l, np.abs(num - gb).max()))
