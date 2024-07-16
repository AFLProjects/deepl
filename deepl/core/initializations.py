from ..utils import time_perf
import autograd.numpy as np


# Initializes an array with zeros given the specified structure.
@time_perf
def zero_init(struct):
    return [np.zeros((struct[i+1], struct[i]+1))
            for i in range(len(struct)-1)]


# Initializes an array with random values uniformly distributed
# between a and b.
@time_perf
def uniform_init(struct, a, b):
    return [np.random.uniform(low=a, high=b, size=(struct[i+1], struct[i]+1))
            for i in range(len(struct)-1)]


# Initializes an array using Xavier (Glorot) initialization.
@time_perf
def xavier_init(struct):
    limit = np.sqrt(6 / (struct[0] + struct[-1]))
    return [np.random.uniform(low=-limit, high=limit,
            size=(struct[i+1], struct[i]+1)) for i in range(len(struct)-1)]


# Initializes an array using He initialization.
@time_perf
def he_init(struct):
    return [np.random.normal(loc=0.0, scale=np.sqrt(1 / struct[0]),
            size=(struct[i+1], struct[i]+1)) for i in range(len(struct)-1)]


# Initializes an array using variance scaling initialization.
@time_perf
def variance_scaling_init(struct):
    return [np.random.normal(loc=0.0, scale=2 / (struct[0] + struct[-1]),
            size=(struct[i+1], struct[i]+1)) for i in range(len(struct)-1)]


# Initializes an array with a constant value c.
@time_perf
def constant_init(struct, c):
    return [np.full((struct[i+1], struct[i]+1), c)
            for i in range(len(struct)-1)]
