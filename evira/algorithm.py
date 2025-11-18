from common.optim import AnnealingOptimiser, QAOAOptimiser
from common.problem import PROBLEMS
from evira.qubo import IndexSelectionQUBO
from common.util import compute_cost

def evira(benefits, weights, budget, rho, t_max, t_conv, epsilon, qaoa_reps, qaoa_shots, mode = 'simulate', type = 'anneal'):
    assert type == 'anneal' or type == 'qaoa', 'select a quantum approach'
    assert mode == 'simulate' or mode == 'quantum', 'select an execution mode'
    # 1. initialise
    z = 0
    lam = 0
    t = 1

    best_xfeas = None
    best_xfeas_benefit = -1
    steps_since_improvement = 0

    # 2a. initial qubo matrix (we don't need to reinitialise each step)
    qubo = IndexSelectionQUBO(benefits, weights, budget, lam, rho, type)
    if type == 'anneal':
        optimiser = AnnealingOptimiser(benefits, weights, budget, mode)
    else:
        optimiser = QAOAOptimiser(benefits, weights, budget, qaoa_reps, qaoa_shots, mode)
    while True:
        print('***** starting iteration', t)
        # 3. compute QUBO matrix
        qubo.update_classical_vals(lam, z)

        x_cost, x_feas, x_cost_weight, x_feas_benefit = optimiser.optimise(qubo.get_qubo())

        print('x_cost:', x_cost, 'weight:', x_cost_weight)
        print('x_feas:', x_feas, 'weight:', compute_cost(x_feas, weights))

        # 6. update z_star
        z = min(0, x_cost_weight - budget)
        print('z:', z)

        # 7. update lambda
        lam = lam + (rho * (x_cost_weight - budget - z))
        print('λ:', lam)

        # 7a. check if found solution is better
        if x_feas_benefit > best_xfeas_benefit:
            best_xfeas = x_feas
            best_xfeas_benefit = x_feas_benefit
            steps_since_improvement = 0
        elif best_xfeas_benefit == -1:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        print('best x_feas is', best_xfeas, 'with benefit', best_xfeas_benefit, 'steps since improvement:', steps_since_improvement)

        # 8. check convergence
        def should_terminate():
            if t > t_max: return True
            if steps_since_improvement > t_conv: return True
            return False
        if should_terminate():
            break

        # 9. update t
        t = t + 1

    # 10. iterate 3-9
    return best_xfeas, best_xfeas_benefit, t
