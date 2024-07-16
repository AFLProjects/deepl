from ..core.initializations import *
from .losses import *
from autograd import grad
from utils import time_perf
                                                  

class SGD_Optimizer:
    @time_perf
    def __init__(self, nn, loss=mse,
                initialization=uniform_init, initialization_args=(0,1), 
                start_learning_rate=0.01, callbacks=[]):
        self.nn = nn  # Neural network to optimize
        self.loss = loss  # Loss function
        self.grad_loss = grad(loss)  # Gradient of the loss function
        self.loss_values = []  # Store loss values for each iteration
        self.initialization = initialization  # Method to initialize weights
        self.initialization_args = initialization_args  # Arguments for initialization
        self.current_w_tensor = initialization(nn.structure,
                                                  *initialization_args)  # Initialize weights
        self.start_learning_rate = start_learning_rate  # Initial learning rate
        self.learning_rate = self.start_learning_rate  # Current learning rate
        self.callbacks = callbacks  # List of callback functions

    def reset(self):
        # Reset weights, loss values, and learning rate
        self.loss_values = []
        self.current_w_tensor = self.initialization(self.nn.structure,
                                                  *self.initialization_args)
        self.learning_rate = self.start_learning_rate
        
    @time_perf
    def train(self, x, y, data_size):
        # Train the neural network using SGD for a specified number of iterations
        for i in range(data_size):
            gradient = self.grad_loss(self.current_w_tensor, self.nn, x[i], y[i])
            loss_value = self.loss(self.current_w_tensor, self.nn, x[i], y[i])
            self.current_w_tensor = self.current_w_tensor - self.learning_rate * gradient
            self.loss_values.append(loss_value)

        return (self.current_w_tensor, self.loss_values)