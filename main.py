from deepl import core
from deepl import training
import matplotlib.pyplot as plt
import autograd.numpy as np

# Example
structure = (2, 2, 2)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])

data_size = 16000
train_x = [np.random.rand(2) for _ in range(data_size)]
train_y = 2 * train_x

validate_size = 128
validate_x = [np.random.rand(2) for _ in range(validate_size)]
validate_y = 2 * validate_x

trainer = training.SGD_Optimizer(nn,
                                loss=training.mse,
                                initialization=core.uniform_init,
                                initialization_args=(0,1), 
                                start_learning_rate=0.1,
                                callbacks=[training.stop_loss_min],
                                callback_args=[(10e-5,)],
                                validate_x=validate_x,
                                validate_y=validate_y)

weights_tensor, loss_values = trainer.train(train_x, train_y, data_size)


# -- Plotting --
plt.figure(figsize=(10, 6))
plt.scatter(range(len(loss_values)), loss_values, label='MSE', s=len(loss_values)*[0.1])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

