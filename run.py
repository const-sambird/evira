import argparse

from common.problem import PROBLEMS
from evira.algorithm import evira
from trummer.algorithm import trummer

def create_arguments():
    parser = argparse.ArgumentParser()

    # hyperparameters - could leave at defaults

    parser.add_argument('-r', '--qaoa-reps', type=int, default=1, help='the number of the repetitions in the QAOA ansatz')
    parser.add_argument('-s', '--qaoa-shots', type=int, default=1024, help='number of shots for the QAOA sampler')
    parser.add_argument('--rho', type=float, default=0.5, help='rho, penalty multiplier')
    parser.add_argument('-t', '--t-max', type=int, default=50, help='t_max, maximum number of iterations')
    parser.add_argument('--t-conv', type=int, default=10, help='t_conv, maximum number of iterations without improvement in x_feas')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='slack variable convergence criterion')
    parser.add_argument('--alpha', type=float, default=4, help='trummer -- multiplier for penalty term alpha')
    parser.add_argument('--num-partitions', type=int, default=3, help='trummer -- index candidate partitions')

    # configuration - set per problem

    parser.add_argument('-q', '--quantum', action='store_true', help='run on real quantum hardware instead of simulating')
    parser.add_argument('-p', '--problem', choices=list(PROBLEMS.keys()), default='I7', help='which index selection problem should we solve?')
    parser.add_argument('algorithm', type=str, choices=['evira', 'trummer'], help='which index selection algorithm to use')
    parser.add_argument('type', type=str, choices=['anneal', 'qaoa'], help='use annealing optimisation or gate-based QAOA?')

    return parser.parse_args()

if __name__ == '__main__':
    args = create_arguments()
    p = PROBLEMS[args.problem]
    if args.algorithm == 'evira':
        xfeas, benefit, steps = evira(p.benefits, p.weights, p.budget, args.rho, args.t_max,
                args.t_conv, args.epsilon, args.qaoa_reps, args.qaoa_shots,
                'quantum' if args.quantum else 'simulate', args.type)
        print('found solution', xfeas, 'with quality', benefit, 'in', steps, 'steps')
    else:
        trummer(p.benefits, p.weights, p.budget, args.num_partitions, args.alpha, args.qaoa_reps,
                args.qaoa_shots, args.type, 'quantum' if args.quantum else 'simulate')
