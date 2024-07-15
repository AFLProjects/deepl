from deepl import core
from deepl import training


import time

import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

import math


grad_loss = grad(training.binary_cross_entropy)


# -- Learning rate --


# Exponential decaying learning rate
def exponential_learning_rate(initial, epoch, decay_rate):
    return initial * math.exp(-decay_rate * epoch)


# -- Tests --

structure = (2, 2, 2)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])
weights_tensor = core.uniform_init(structure, 0, 1)

alpha_initial = 0.01
alpha = alpha_initial
mae = 10e9
loss_values = []
iterations = 5000
for i in range(iterations):
    input = np.random.rand(2)
    output_expected = 2 * input
    mae = training.binary_cross_entropy(weights_tensor, nn, input, output_expected)
    loss_values.append(mae)
    gradient_tensor = grad_loss(weights_tensor, nn, input, output_expected)
    weights_tensor = weights_tensor - alpha * gradient_tensor
    alpha = exponential_learning_rate(alpha_initial, i, 0.0)

print(f'mae : {mae}')
print(f'w : {weights_tensor}')


# -- Plotting --


plt.figure(figsize=(10, 6))
plt.scatter(range(iterations), loss_values, label='MAE', s=iterations*[0.1])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()
