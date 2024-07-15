import autograd.numpy as np

# Initializes an array with zeros given the specified structure.
def zero_init(structure):
    return np.zeros(structure)

# Initializes an array with random values uniformly distributed between a and b.
def uniform_init(structure, a, b):
    return np.random.uniform(low=a, high=b, size=structure)

# Initializes an array using Xavier (Glorot) initialization.
def xavier_init(structure):
    limit = np.sqrt(6 / (structure[0] + structure[-1]))
    return np.random.uniform(low=-limit, high=limit, size=structure)

# Initializes an array using He initialization.
def he_init(structure):
    return np.random.normal(loc=0.0, scale=np.sqrt(1 / structure[0]), size=structure)

# Initializes an array using variance scaling initialization.
def variance_scaling_init(structure):
    return np.random.normal(loc=0.0, scale=2 / (structure[0] + structure[-1]), size=structure)

# Initializes an array with a constant value c.
def constant_init(structure, c):
    return np.full(structure, c)