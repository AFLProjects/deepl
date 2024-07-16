from deepl import core
from deepl import training
from deepl import visualization
import autograd.numpy as np

# Example

# Define structure
structure = (3, 3, 3)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])

# Training data
data_size = 16000
train_x = [np.random.rand(3) for _ in range(data_size)]
train_y = 100 * train_x

# Validation data
validate_size = 64
validate_x = [np.random.rand(3) for _ in range(validate_size)]
validate_y = 100 * validate_x

# Optimizer and parameters
trainer = training.SGD_Optimizer(nn,
                                 loss=training.mse,
                                 init=core.uniform_init,
                                 init_args=(0, 1),
                                 start_lr=0.1,
                                 callbacks=[training.stop_loss_min,
                                            training.checkpoints],
                                 callback_args=[(10e-3,),
                                                (250,)],
                                 validate_x=validate_x,
                                 validate_y=validate_y,
                                 reg=training.lasso,
                                 reg_params=(0.001,))

# Train
weights_tensor, loss_values = trainer.train(train_x, train_y, data_size)

# Plots
visualization.loss_plot(trainer, 0.5, 'MSE')
visualization.mean_validation_plot(trainer, 0.5, 'MSE')
