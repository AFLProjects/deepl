import autograd.numpy as np
from utils import time_perf

# Represents a single node in a dense (fully connected) layer.
class Dense_Node:

    def __init__(self, activation, input_size):
        # Initialize with activation function and input size.
        self.activation = activation
        self.input_size = input_size

    def output(self, input, weights):
        # Compute the node's output using the activation function.
        return self.activation(np.dot(input, weights))

# Represents a dense (fully connected) layer.
class Dense_Layer:

    def __init__(self, activation_function, layer_size, input_size):
        # Initialize with activation function, layer size, and input size.
        self.layer_size = layer_size
        self.input_size = input_size
        self.nodes = [Dense_Node(activation_function, input_size) for _ in range(layer_size)]

    def output(self, input, weights_matrix):
        # Compute the layer's output.
        return np.array([self.nodes[i].output(input, weights_matrix[i]) for i in range(len(self.nodes))])

# Represents an artificial neural network composed of dense layers.
class ANN:

    @time_perf
    def __init__(self, sizes, activation_list):
        # Initialize the ANN with layer sizes and activation functions.
        self.layers = [Dense_Layer(activation_list[i-1], sizes[i], sizes[i-1]) for i in range(1, len(sizes))]

    def output(self, input, weights_tensor):
        # Compute the ANN's output.
        current = input
        for i in range(len(self.layers)):
            current = self.layers[i].output(current, weights_tensor[i])
        return current
