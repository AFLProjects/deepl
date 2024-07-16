import autograd.numpy as np


# Check if the mean validation loss is below a specified threshold
def stop_loss_min(opt, stop_loss_min, epoch):
    mean_loss = np.mean([opt.loss(opt.current_w_tensor, opt.nn, x,
                                  y, opt.reg, opt.reg_params)
                        for x, y in zip(opt.validate_x, opt.validate_y)])
    opt_new = opt
    opt_new.mean_validation_loss.append(mean_loss)
    return (opt_new,
            mean_loss <= stop_loss_min,
            False)


# Adjust the learning rate according to a specified schedule
def lr_scheduler(opt, schedule, epoch):
    opt_new = opt
    opt_new.lr = schedule(epoch)
    return (opt_new,
            False,
            False)


# Save model checkpoints at specified intervals
def checkpoints(opt, rate, epoch):
    return (opt,
            False,
            epoch % rate == 0)
