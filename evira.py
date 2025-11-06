import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler
from dimod import ExactSolver

from problem import Problem, QIA_PROBLEMS
from qubo import IndexSelectionQUBO, AnnealerQUBO

def get_solver(is_quantum: bool):
    if is_quantum:
        return EmbeddingComposite(DWaveSampler())
    else:
        return PathIntegralAnnealingSampler()

def compute_cost(x, weights):
    return sum([x[i] * weights[i] for i in range(len(x))])

def compute_benefit(x, benefits):
    return sum([x[i] * benefits[i] for i in range(len(x))])

def admm(benefits, weights, budget, rho, t_max, t_conv, epsilon, num_reads = 100):
    # 1. initialise
    z = 0
    lam = 0
    t = 1

    best_xfeas = None
    best_xfeas_benefit = -1
    steps_since_improvement = 0

    # 2. apply embedding
    solver = get_solver(False)

    # 2a. initial qubo matrix (we don't need to reinitialise each step)
    qubo = IndexSelectionQUBO(benefits, weights, budget, lam, rho)
    while True:
        print('***** starting iteration', t)
        # 3. compute QUBO matrix
        qubo.update_classical_vals(lam, z)

        # 4. anneal
        solution = solver.sample_qubo(qubo.get_qubo(), num_reads=num_reads).record

        # 5. compute x_cost, x_feas using samples
        x_cost = solution[np.recarray.argmin(solution.energy)]
        x_costs = np.asarray([compute_cost(np.asarray(sample, dtype=np.float32), weights) for sample in solution.sample])
        x_benefits = np.asarray([compute_benefit(np.asarray(sample, dtype=np.float32), benefits) for sample in solution.sample])
        x_benefits = np.where(x_costs <= budget, x_benefits, float('-inf'))
        x_feas = solution[np.argmax(x_benefits)]

        x_cost_weight = compute_cost(np.asarray(x_cost.sample, dtype=np.float32), weights)
        x_feas_weight = compute_cost(np.asarray(x_feas.sample, dtype=np.float32), weights)
        x_feas_benefit = np.max(x_benefits)

        print('x_cost:', x_cost, 'weight:', x_cost_weight)
        print('x_feas:', x_feas, 'weight:', x_feas_weight)

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
    return best_xfeas

if __name__ == '__main__':
    p = QIA_PROBLEMS['CDB_I7_ADJ']
    x = admm(p.benefits, p.weights, p.budget, 0.5, 50, 10, 1e-3)
    print(x)