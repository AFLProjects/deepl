from .optimizers import SGD_Optimizer
import autograd.numpy as np

def stop_loss_min(opt, stop_loss_min):
    mean_loss = np.mean([opt.loss(opt.current_w_tensor, opt.nn, x, y) 
                        for x,y in zip(opt.validate_x, opt.validate_y)])
    return (mean_loss <= stop_loss_min)