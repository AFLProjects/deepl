from ..core.initializations import *
from .losses import *
from autograd import grad
                                                  

class SGD_Optimizer:
    def __init__(self, nn, loss=mse,
                initialization=uniform_init, initialization_args=(0,1), 
                start_learning_rate=0.01, callbacks=[]):
        self.nn = nn
        self.loss = loss 
        self.grad_loss = grad(loss)
        self.loss_values = []
        self.initialization = initialization
        self.initialization_args = initialization_args
        self.current_w_tensor = initialization(nn.structure,
                                                  *initialization_args)
        self.start_learning_rate = start_learning_rate
        self.learning_rate = self.start_learning_rate
        self.callbacks = callbacks

    def reset(self):
        self.loss_values = []
        self.current_w_tensor = self.initialization(self.nn.structure,
                                                  *self.initialization_args)
        self.learning_rate = self.start_learning_rate
        

    def train(self, dataset, data_size, batch_size=32):
        # Train the neural network using SGD for a specified number of iterations
        num_batches = data_size // batch_size
        for batch_index in range(num_batches):
            batch_start = batch_index * batch_size
            batch_end = batch_start + batch_size
            batch_x = dataset[0][batch_start:batch_end]
            batch_y = dataset[1][batch_start:batch_end]

            # Compute gradients and update weights
            gradients = np.mean([self.grad_loss(self.current_w_tensor, self.nn, x, y) for x, y in zip(batch_x, batch_y)], axis=0)
            loss_values_batch = [self.loss(self.current_w_tensor, self.nn, x, y) for x, y in zip(batch_x, batch_y)]
            self.current_w_tensor -= self.learning_rate * gradients
            self.loss_values.extend(loss_values_batch)

            # Execute callbacks if any
            for callback in self.callbacks:
                callback(self)

        return self.current_w_tensor, self.loss_values