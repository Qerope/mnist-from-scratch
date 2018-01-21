# mnist-from-scratch

Fully connected network for MNIST written with numpy only.

## Setup

Download the four `*-ubyte.gz` files from yann.lecun.com/exdb/mnist/ into
`data/`, then

    pip install -r requirements.txt
    python train.py

## Results

784-100-10, sigmoid hidden layer, softmax output, cross-entropy, batch size
32, eta 0.5, 30 epochs: about 97% test accuracy. ReLU with eta 0.1 gets to
about the same in fewer epochs.

`gradcheck.py` compares backprop against finite differences; it should print
errors below 1e-7.
