import autograd.numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Hyperbolic tangent activation function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# Rectified Linear Unit (ReLU) activation function
def relu(x):
    return np.maximum(0, x)


# Leaky Rectified Linear Unit (Leaky ReLU) activation function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# Exponential Linear Unit (ELU) activation function
def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# Swish activation function
def swish(x):
    return x * sigmoid(x)


# Fixed-point linear activation function (Identity function)
def fixed_point(x):
    return x
