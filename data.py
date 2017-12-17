import gzip
import struct

import numpy as np


def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        ndim = magic & 0xff
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load(dirname='data'):
    Xtr = _read_idx(dirname + '/train-images-idx3-ubyte.gz').reshape(-1, 784) / 255.0
    ytr = _read_idx(dirname + '/train-labels-idx1-ubyte.gz')
    Xte = _read_idx(dirname + '/t10k-images-idx3-ubyte.gz').reshape(-1, 784) / 255.0
    yte = _read_idx(dirname + '/t10k-labels-idx1-ubyte.gz')
    return Xtr, ytr, Xte, yte


def one_hot(y, k=10):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out
