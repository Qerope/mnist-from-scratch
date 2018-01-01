import data
from network import Network

Xtr, ytr, Xte, yte = data.load()
net = Network([784, 100, 10])
history = net.sgd(Xtr, data.one_hot(ytr), epochs=30, batch_size=32, eta=0.5,
                  lmbda=5.0, test=(Xte, yte))
net.save('weights.npz')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(range(1, len(history) + 1), history)
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.savefig('accuracy.png')
