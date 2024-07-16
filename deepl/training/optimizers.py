from ..core.initializations import uniform_init
from ..utils import time_perf
from .losses import mse
from autograd import grad
import autograd.numpy as np


class SGD_Optimizer:

    @time_perf
    def __init__(self, nn, loss=mse,
                 init=uniform_init,
                 init_args=(0, 1),
                 start_lr=0.01,
                 callbacks=[],
                 callback_args=[],
                 validate_x=[],
                 validate_y=[]):
        self.nn = nn  # Neural network to optimize
        self.loss = loss  # Loss function
        self.grad_loss = grad(loss)  # Gradient of the loss function
        self.loss_values = []  # Store loss values for each iteration
        self.init = init  # Method to initialize weights
        self.init_args = init_args  # Arguments for initialization
        # Initialize weights
        self.current_w_tensor = init(nn.structure, *init_args)
        self.start_lr = start_lr  # Initial learning rate
        self.lr = self.start_lr  # Current learning rate
        self.callbacks = callbacks  # List of callback functions
        # List of arguments for callback functions
        self.callback_args = callback_args
        self.validate_x = validate_x  # Set of inputs to validate
        self.validate_y = validate_y  # Set of outputs to validate
        # List for tracking weights tensor at regular intervals
        self.checkpoints_tracking = []  # Array for checkpoint callback
        self.mean_validation_loss = []  # Array for min loss callback

    # Train the neural network using SGD for a specified number of iterations
    @time_perf
    def train(self, x, y, data_size):

        stop_event, checkpoint_event = False, False

        for i in range(data_size):

            if stop_event:
                break

            if checkpoint_event:
                self.checkpoints_tracking.append(self.current_w_tensor)
                checkpoint_event = False

            gradient = self.grad_loss(self.current_w_tensor,
                                      self.nn, x[i], y[i])
            loss_value = self.loss(self.current_w_tensor,
                                   self.nn, x[i], y[i])
            
            new_tensor = [w - self.lr * g for w,g in zip(self.current_w_tensor, gradient)]
            self.current_w_tensor = new_tensor
            self.loss_values.append(loss_value)

            for callback, callback_arg in zip(self.callbacks,
                                              self.callback_args):
                self, stop_call, checkpoint_call = callback(self,
                                                            *callback_arg, i)
                stop_event |= stop_call
                checkpoint_event |= checkpoint_call

        return (self.current_w_tensor, self.loss_values)
