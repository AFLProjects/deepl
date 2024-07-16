import autograd.numpy as np


# Calculates the mean squared error (MSE) between the network output
# and correct output
def mse(weights_tensor, nn, input, correct_output, reg, reg_params):
    output = nn.output(input, weights_tensor)
    return np.mean((output - correct_output) ** 2) + reg(weights_tensor, *reg_params)


# Calculates the mean absolute error (MAE) between the network
# output and correct output
def mae(weights_tensor, nn, input, correct_output, reg, reg_params):
    output = nn.output(input, weights_tensor)
    return np.mean(np.abs(output - correct_output)) + reg(weights_tensor, *reg_params)


# Calculates the binary cross-entropy loss between the network output
# and correct output, Added clipping for numerical stability
def binary_cross_entropy(weights_tensor, nn, input, correct_output, reg, reg_params):
    output = nn.output(input, weights_tensor)
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.mean(correct_output * np.log(output) +
                    (1 - correct_output) * np.log(1 - output)) + reg(weights_tensor, *reg_params)


# Calculates the hinge loss between the network output and correct output
def hinge_loss(weights_tensor, nn, input, correct_output, reg, reg_params):
    output = nn.output(input, weights_tensor)
    return np.mean(np.maximum(0, 1 - output * correct_output)) + reg(weights_tensor, *reg_params)
