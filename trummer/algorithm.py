from math import log2, ceil

from trummer.qubo import TrummerQUBO
from common.optim import QAOAOptimiser, AnnealingOptimiser
from common.problem import PROBLEMS
from common.util import compute_cost

def trummer(benefits, weights, budget, num_partitions, lam, qaoa_reps, qaoa_shots, type, mode = 'simulate'):
    SPACE_CONSUMPTION_QUBITS = ceil(log2(budget))

    qubo = TrummerQUBO(benefits, weights, budget, SPACE_CONSUMPTION_QUBITS, lam, num_partitions, type)

    if type == 'anneal':
        optimiser = AnnealingOptimiser(benefits, weights, budget, mode)
    else:
        optimiser = QAOAOptimiser(benefits, weights, budget, qaoa_reps, qaoa_shots, mode)
    
    x_cost, x_feas, x_cost_weight, x_feas_benefit = optimiser.optimise(qubo.get_qubo())

    print('x_cost:', x_cost, 'weight:', x_cost_weight)
    print('x_feas:', x_feas, 'weight:', compute_cost(x_feas, weights))
    print('best x_feas is', x_feas, 'with benefit', x_feas_benefit)
