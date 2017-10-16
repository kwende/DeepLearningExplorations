import numpy as np

class Network(object):
    """
    Describes the network. 
    Use: net = Network([1, 10, 2]) 
        where sizes = [1, 10, 2]
    """

    def __init__(self, sizes):
        # number of layers (including input and output)
        self.num_layers = len(sizes)
        # the size array.
        self.sizes = sizes
        # randomize the biases for each node for
        # each layer but the input layer.
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        # build a list of random numbers for the weights
        # the weights are setup so that A nodes in layer n
        # connects to B nodes in layer n+1.
        self.weights = [np.random.randn(x,y) for x, y in zip(sizes[:-1],sizes[1:])]

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def feedforward(self, a):
        for b, w in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a