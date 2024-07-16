import autograd.numpy as np

def stop_loss_min(opt, stop_loss_min, epoch):
    # Check if the mean validation loss is below a specified threshold
    mean_loss = np.mean([opt.loss(opt.current_w_tensor, opt.nn, x, y) 
                        for x, y in zip(opt.validate_x, opt.validate_y)])
    opt_new = opt
    opt_new.mean_validation_loss.append(mean_loss)
    return (opt_new,
            mean_loss <= stop_loss_min,
            False)

def lr_scheduler(opt, schedule, epoch):
    # Adjust the learning rate according to a specified schedule
    opt_new = opt
    opt_new.lr = schedule(epoch)
    return (opt_new,
            False,
            False)

def checkpoints(opt, rate, epoch):
    # Save model checkpoints at specified intervals
    return (opt, 
            False, 
            epoch % rate == 0)
