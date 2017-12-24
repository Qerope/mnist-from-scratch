import data
from network import Network

Xtr, ytr, Xte, yte = data.load()
net = Network([784, 100, 10])
net.sgd(Xtr, data.one_hot(ytr), epochs=30, batch_size=32, eta=0.5, test=(Xte, yte))
