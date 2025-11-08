import argparse

from optim import AnnealingOptimiser, QAOAOptimiser
from problem import PROBLEMS
from qubo import IndexSelectionQUBO
from util import compute_cost

def admm(benefits, weights, budget, rho, t_max, t_conv, epsilon, qaoa_reps, qaoa_shots, mode = 'simulate', type = 'anneal'):
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

def create_arguments():
    parser = argparse.ArgumentParser()

    # hyperparameters - could leave at defaults

    parser.add_argument('-r', '--qaoa-reps', type=int, default=1, help='the number of the repetitions in the QAOA ansatz')
    parser.add_argument('-s', '--qaoa-shots', type=int, default=1024, help='number of shots for the QAOA sampler')
    parser.add_argument('--rho', type=float, default=0.5, help='rho, penalty multiplier')
    parser.add_argument('-t', '--t-max', type=int, default=50, help='t_max, maximum number of iterations')
    parser.add_argument('--t-conv', type=int, default=10, help='t_conv, maximum number of iterations without improvement in x_feas')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='slack variable convergence criterion')

    # configuration - set per problem

    parser.add_argument('-q', '--quantum', action='store_true', help='run on real quantum hardware instead of simulating')
    parser.add_argument('-p', '--problem', choices=list(PROBLEMS.keys()), default='I7', help='which index selection problem should we solve?')
    parser.add_argument('type', type=str, choices=['anneal', 'qaoa'], help='use annealing optimisation or gate-based QAOA?')

    return parser.parse_args()

if __name__ == '__main__':
    args = create_arguments()
    p = PROBLEMS[args.problem]
    xfeas, benefit, steps = admm(p.benefits, p.weights, p.budget, args.rho, args.t_max,
             args.t_conv, args.epsilon, args.qaoa_reps, args.qaoa_shots,
             'quantum' if args.quantum else 'simulate', args.type)
    print('found solution', xfeas, 'with quality', benefit, 'in', steps, 'steps')