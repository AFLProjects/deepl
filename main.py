from deepl import core
from deepl import training
import matplotlib.pyplot as plt
import autograd.numpy as np

# Example
structure = (2, 2, 2)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])

N = 5000
data_inputs = [np.random.rand(2) for _ in range(N)]
data_outputs = 2 * data_inputs
dataset = [data_inputs, data_outputs]

trainer = training.SGD_Optimizer(nn,
                                loss=training.mse,
                                initialization=core.uniform_init,
                                initialization_args=(0,1), 
                                start_learning_rate=0.01,
                                callbacks=[])

weights_tensor, loss_values = trainer.train(dataset, N, batch_size=1)


# -- Plotting --
plt.figure(figsize=(10, 6))
plt.scatter(range(len(loss_values)), loss_values, label='MSE', s=len(loss_values)*[0.1])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

