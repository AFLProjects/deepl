from ..utils import time_perf
import autograd.numpy as np


# Represents a dense (fully connected) layer.
class Dense_Layer:

    # Initialize with activation function, layer size, and input size.
    def __init__(self, activation_function, layer_size, input_size):
        self.activation = activation_function
        self.layer_size = layer_size
        self.input_size = input_size

    # Compute the layer's output.
    def output(self, input, weights_matrix):
        return self.activation(np.dot(weights_matrix, input))


# Represents an artificial neural network composed of dense layers.
class Dense_ANN:

    # Initialize the ANN with layer sizes and activation functions.
    @time_perf
    def __init__(self, sizes, activation_list):
        self.layers = [Dense_Layer(activation_list[i-1], sizes[i], sizes[i-1])
                       for i in range(1, len(sizes))]
        self.structure = sizes
        self.activation_list = activation_list

    # Compute the ANN's output.s
    def output(self, input, weights_tensor):
        next = input
        for i in range(len(self.layers)):
            next = self.layers[i].output(next, weights_tensor[i])
        return next
