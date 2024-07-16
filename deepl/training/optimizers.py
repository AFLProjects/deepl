from ..core.initializations import *
from .losses import *
from autograd import grad
                                                  

class SGD_Optimizer:
    def __init__(self, nn, loss=mse,
                initialization=uniform_init, initialization_args=(0,1), 
                start_learning_rate=0.01, callbacks=[]):
        # set neural netowkr
        self.nn = nn
        
        # loss params
        self.loss = loss 
        self.grad_loss = grad(loss)
        self.loss_values = []
        
        # set initialization params and initialize tensor
        self.initialization = initialization
        self.initialization_args = initialization_args
        self.start_w_tensor = initialization(nn.structure, *initialization_args)
        self.current_w_tensor = self.start_w_tensor
        
        # set learning rate
        self.start_learning_rate = start_learning_rate
        self.learning_rate = self.start_learning_rate
        
        # set callbacks
        self.callbacks = callbacks

    def reset(self):
        self.start_w_tensor = self.initialization(self.nn.structure,
                                                  *self.initialization_args)
        self.current_w_tensor = self.start_w_tensor
        self.learning_rate = self.start_learning_rate

    def train(self, dataset, data_size):
        for i in range(data_size):
            gradient = self.grad_loss(self.current_w_tensor, self.nn,
                                     dataset[0][i], dataset[1][i])
            loss_value = self.loss(self.current_w_tensor, self.nn,
                                    dataset[0][i], dataset[1][i])
            self.current_w_tensor = self.current_w_tensor - self.learning_rate * gradient
            self.loss_values.append(loss_value)
        return (self.current_w_tensor, self.loss_values)