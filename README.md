# deepl

A small deep learning library for training end-to-end Artificial Neural Networks (ANNs), primarily based on `numpy` and `autograd`. (This is a very small 2-day project for learning purposes.)

## Features

### Network Structure
- `Dense_ANN`
- `Dense_Layer`

### Weights Initialization Models
- `zero_init`
- `uniform_init`
- `xavier_init`
- `he_init`
- `variance_scaling_init`
- `constant_init`

### Activation Functions
- `sigmoid`
- `tanh`
- `relu`
- `leaky_relu`
- `elu`
- `swish`
- `fixed_point`

### Output Evaluation and Loss Functions
- `mse` (Mean Squared Error)
- `mae` (Mean Absolute Error)
- `binary_cross_entropy`
- `hinge_loss`

### Optimizer
- `SGD_Optimizer` (Stochastic Gradient Descent) capable of training fully connected ANNs.

### Callbacks
- `stop_loss_min`
- `lr_scheduler`
- `checkpoints`

### Performance Visualization
- `loss_plot`
- `mean_validation_plot`

## Limitations
- Only supports networks of the form `(N, N, N, N, ...)`.
- Biases and regularizers are not implemented.
- Uses mostly `numpy` with `autograd`, which does not take advantage of GPU and parallelism as modern libraries would.
- Only implements end-to-end training, not model-based deep learning.

## Future Improvements
- Support for networks of the form `(A, N, ..., N, B)`.
- Implementation of biases and regularizers.
- Possibly custom tensors and a slightly modified `autograd`.

## Example

```python
from deepl import core
from deepl import training
from deepl import visualization
import autograd.numpy as np

# Example
# Structure
structure = (2, 2, 2)
nn = core.Dense_ANN(structure, [core.relu, core.fixed_point])

# Training data
data_size = 16000
train_x = [np.random.rand(2) for _ in range(data_size)]
train_y = 2 * train_x

# Validation data
validate_size = 64
validate_x = [np.random.rand(2) for _ in range(validate_size)]
validate_y = 2 * validate_x

# Optimiser
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

# Train
weights_tensor, loss_values = trainer.train(train_x, train_y, data_size)

# Plot
visualization.loss_plot(trainer, 0.5, 'MSE')
visualization.mean_validation_plot(trainer, 0.5, 'MSE')
```

<div style="display: flex; justify-content: space-between;">
    <img src="image/img1.png" alt="Image 1" style="width: 48%;">
    <img src="image/img2.png" alt="Image 2" style="width: 48%;">
</div>




