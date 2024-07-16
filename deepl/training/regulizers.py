import autograd.numpy as np

def empty_regulizer(w_tensor):
    return 0

def lasso(w_tensor, alpha):
    return alpha * np.sum([np.sum(np.abs(mat)) for mat in w_tensor])

def ridge(w_tensor, alpha):
    return alpha * np.sum([np.sum(mat ** 2) for mat in w_tensor])

def elastic_net(w_tensor, alpha1, alpha2):
    return lasso(w_tensor, alpha1) + ridge(w_tensor, alpha2)