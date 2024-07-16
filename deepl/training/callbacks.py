from autograd import grad
from optimizers import train_sgd_step


"""
def train_sgd(training_data, weights_tensor_init, nn, loss_fnc,
               stop_loss=10e-5, learning_rate=0.01, min_iter_stop=5000):
    grad_loss = grad(loss_fnc)
    loss_value = 10e4
    loss_values = []
    weights_tensor = weights_tensor_init
    for i in range(len(training_data)):
        params = (weights_tensor, training_data[0][i], training_data[1][i],
                  learning_rate, loss_fnc, grad_loss, nn)
        weights_tensor, loss_value = train_sgd_step(params)
        loss_values.append(loss_value)
        if loss_value < stop_loss and i >= min_iter_stop:
            break
        # use callback instead
    return (weights_tensor, loss_value, loss_values)

def callback1_test():
    return None
""""""


        


