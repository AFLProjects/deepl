import autograd.numpy as np


# Returns 0, representing no regularization.
def empty_regulizer(w_tensor):
    return 0


# Computes L1 regularization (Lasso) for a weight tensor.
def lasso(w_tensor, alpha):
    return alpha * np.sum([np.sum(np.abs(mat)) for mat in w_tensor])


# Computes L2 regularization (Ridge) for a weight tensor.
def ridge(w_tensor, alpha):
    return alpha * np.sum([np.sum(mat ** 2) for mat in w_tensor])


# Combines L1 and L2 regularization (Elastic Net) for a weight tensor.
def elastic_net(w_tensor, alpha1, alpha2):
    return lasso(w_tensor, alpha1) + ridge(w_tensor, alpha2)
