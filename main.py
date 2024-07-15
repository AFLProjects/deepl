from deepl import core

import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

import math


# -- ANN Structure --


# -- Loss functions --


# ANN Loss
def mse_loss(weights_tensor, nn, input, expected_output):
    output = nn.output(input, weights_tensor)
    diff = output - expected_output
    return np.mean(diff ** 2)


# -- Gradient --


grad_loss = grad(mse_loss)





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



structure = (2, 2, 2)
nn = core.ANN(structure, [core.relu, core.fixed_point])


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
