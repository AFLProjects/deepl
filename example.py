from deepl import core
from deepl import training
from deepl import visualization
import autograd.numpy as np

# Example
structure = (2, 2, 2)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])

data_size = 16000
train_x = [np.random.rand(2) for _ in range(data_size)]
train_y = 2 * train_x

validate_size = 64
validate_x = [np.random.rand(2) for _ in range(validate_size)]
validate_y = 2 * validate_x

"""
def lr_schedule(epoch):
    epoch_mod = epoch % 1000
    if 0 <= epoch_mod <= 500:
        return 0.1
    else:
        return 0.01
"""

trainer = training.SGD_Optimizer(nn,
                                 loss=training.mse,
                                 init=core.uniform_init,
                                 init_args=(0, 1),
                                 start_lr=0.1,
                                 callbacks=[training.stop_loss_min,
                                            training.checkpoints],
                                 callback_args=[(10e-4,),
                                                (250,)],
                                 validate_x=validate_x,
                                 validate_y=validate_y)

weights_tensor, loss_values = trainer.train(train_x, train_y, data_size)

visualization.loss_plot(trainer, 0.5, 'MSE')
visualization.mean_validation_plot(trainer, 0.5, 'MSE')
