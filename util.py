def compute_cost(x, weights):
    return sum([x[i] * weights[i] for i in range(len(x))])

def compute_benefit(x, benefits):
    return sum([x[i] * benefits[i] for i in range(len(x))])
