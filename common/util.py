import numpy as np

def compute_cost(x, weights):
    x = np.asarray(x, dtype=np.float64)
    return sum([x[i] * weights[i] for i in range(len(weights))])

def compute_benefit(x, benefits):
    x = np.asarray(x, dtype=np.float64)
    return sum([x[i] * benefits[i] for i in range(len(benefits))])
