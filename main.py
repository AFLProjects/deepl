from deepl import core

import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

import math

core.test_function()

# -- ANN Structure --


class Dense_Node:

    def __init__(self, activation, input_size):
        self.activation = activation
        self.input_size = input_size

    def output(self, input, weights):
        return self.activation(np.dot(input, weights))


class Dense_Layer:

    def __init__(self, activation_function, layer_size, input_size):
        self.layer_size = layer_size
        self.input_size = input_size
        self.nodes = [Dense_Node(activation_function, input_size)
                      for i in range(layer_size)]

    def output(self, input, weights_matrix):
        return np.array([self.nodes[i].output(input, weights_matrix[i])
                         for i in range(len(self.nodes))])


class ANN:

    def __init__(self, structure, activation_list):  # structure="3:2:1"
        strs = structure.split(':')
        sizes = [int(strs[i]) for i in range(len(strs))]
        self.layers = []
        for i in range(1, len(sizes)):
            layer = Dense_Layer(activation_list[i-1], sizes[i], sizes[i-1])
            self.layers.append(layer)

    def output(self, input, weights_tensor):
        current = input
        for i in range(len(self.layers)):
            current = self.layers[i].output(current, weights_tensor[i])
        return current


# -- Loss functions --


# ANN Loss
def mse_loss(weights_tensor, nn, input, expected_output):
    output = nn.output(input, weights_tensor)
    diff = output - expected_output
    return np.mean(diff ** 2)


grad_loss = grad(mse_loss)


# -- Activation funtions --


# Relu activation
def relu(x):
    return np.maximum(0, x)


# Identity activation function
def iden(x):
    return x


# -- Weight initialization --


# Random weight initialization
weights_tensor = np.random.rand(2, 2, 2)

# Zero weight initialization
# weights_tensor = np.zeros((2, 2, 2))

# Xavier weight initialization
n_in, n_hidden, n_out = 2, 2, 2
# limit = np.sqrt(6 / (n_in + n_out))
# weights_tensor = np.random.uniform(-limit, limit,
#                                    size=(n_in, n_hidden, n_out))


# -- Learning rate --


# Exponential decaying learning rate
def exponential_learning_rate(initial, epoch, decay_rate):
    return initial * math.exp(-decay_rate * epoch)


# -- Tests --


nn = ANN("2:2:2", [relu, iden])


alpha_initial = 0.01
alpha = alpha_initial
mse = 10e9
loss_values = []
iterations = 5000
for i in range(iterations):
    input = np.random.rand(2)
    output_expected = 2 * input
    mse = mse_loss(weights_tensor, nn, input, output_expected)
    loss_values.append(mse)
    gradient_tensor = grad_loss(weights_tensor, nn, input, output_expected)
    weights_tensor = weights_tensor - alpha * gradient_tensor
    alpha = exponential_learning_rate(alpha_initial, i, 0.0)

print(f'mse : {mse}')
print(f'w : {weights_tensor}')

# -- Plotting --

plt.figure(figsize=(10, 6))
plt.scatter(range(iterations), loss_values, label='MSE', s=iterations*[0.1])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()
