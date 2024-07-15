import autograd.numpy as np
from utils import time_perf

# Represents a dense (fully connected) layer.
class Dense_Layer:

    def __init__(self, activation_function, layer_size, input_size):
        # Initialize with activation function, layer size, and input size.
        self.activation = activation_function
        self.layer_size = layer_size
        self.input_size = input_size

    def output(self, input, weights_matrix):
        # Compute the layer's output.
        return self.activation(np.dot(weights_matrix, input))

# Represents an artificial neural network composed of dense layers.
class Dense_ANN:

    @time_perf
    def __init__(self, sizes, activation_list):
        # Initialize the ANN with layer sizes and activation functions.
        self.layers = [Dense_Layer(activation_list[i-1], sizes[i], sizes[i-1])
                       for i in range(1, len(sizes))]

    def output(self, input, weights_tensor):
        # Compute the ANN's output.s
        next = input
        for i in range(len(self.layers)):
            next = self.layers[i].output(next, weights_tensor[i])
        return next
