import numpy as np
from scipy.optimize import minimize

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import PathIntegralAnnealingSampler
from dimod import ExactSolver

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from util import compute_benefit, compute_cost

class AnnealingOptimiser:
    def __init__(self, benefits, weights, budget, mode = 'simulate', num_reads = 100):
        assert mode == 'simulate' or mode == 'quantum' or mode == 'exact', 'select a supported solver'
        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.mode = mode
        self.num_reads = num_reads
        self.solver = self._get_solver()
    
    def _get_solver(self):
        if self.mode == 'simulate':
            return PathIntegralAnnealingSampler()
        if self.mode == 'quantum':
            return EmbeddingComposite(DWaveSampler())
        if self.mode == 'exact':
            return ExactSolver()

    def optimise(self, qubo) -> tuple[tuple, tuple, float]:
        '''
        Perform the optimisation

        argmax E_aug [x, λ, z]
          x

        With respect to x.

        :param qubo:             the QUBO (in the form D-Wave/dimod expects)
        :returns x_cost:         the set of binary variables with the lowest computed cost
        :returns x_feas:         the set of binary variables that is the most beneficial feasible solution
        :returns x_cost_weight:  the storage weight consumed by x_cost
        :returns x_feas_benefit: the benefit computed by x_feas
        '''
        # 4. anneal
        solution = self.solver.sample_qubo(qubo, num_reads=self.num_reads).record

        # 5. compute x_cost, x_feas using samples
        x_cost = solution[np.recarray.argmin(solution.energy)]
        x_costs = np.asarray([compute_cost(np.asarray(sample, dtype=np.float32), self.weights) for sample in solution.sample])
        x_benefits = np.asarray([compute_benefit(np.asarray(sample, dtype=np.float32), self.benefits) for sample in solution.sample])
        x_benefits = np.where(x_costs <= self.budget, x_benefits, float('-inf'))
        x_feas = solution[np.argmax(x_benefits)]

        x_cost_weight = compute_cost(np.asarray(x_cost.sample, dtype=np.float32), self.weights)
        x_feas_weight = compute_cost(np.asarray(x_feas.sample, dtype=np.float32), self.weights)
        x_feas_benefit = np.max(x_benefits)

        return x_cost.sample, x_feas.sample, x_cost_weight, x_feas_benefit

class QAOAOptimiser:
    def __init__(self, benefits, weights, budget, reps = 1, shots = 1024, mode = 'simulate'):
        assert mode == 'simulate' or mode == 'quantum', 'select a supported solver'
        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.reps = reps
        self.shots = shots
        self.mode = mode
        self.n_indexes = len(benefits)

        if mode == 'quantum':
            self.service = QiskitRuntimeService()
            self.backend = self.service.least_busy(
                operational=True, simulator=False, min_num_qubits=127
            )
        else:
            self.backend = AerSimulator()
        
        print('initialised qiskit connection with backend', self.backend)

        self.pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
    
    def optimise(self, qubo):
        sampler = Sampler(mode=self.backend, options={'default_shots': self.shots})
        classical_optimiser = COBYLA()

        qaoa = QAOA(sampler=sampler, optimizer=classical_optimiser, reps=self.reps, initial_point=None, transpiler=self.pass_manager)

        operator, offset = qubo
        qaoa._check_operator_ansatz(operator)
        operator = operator.apply_layout(qaoa.ansatz.layout)

        result = qaoa.compute_minimum_eigenvalue(operator)

        solution = []
        
        for bits, freq in result.eigenstate.items():
            b = list(bits)
            b.reverse()
            b = [int(bit) for bit in b]
            solution.append((b, -freq))
        
        solution = np.array(solution, dtype=[('sample', 'i1', (self.n_indexes,)), ('energy', '<f8')])
        solution = solution.view(np.recarray)

        # 5. compute x_cost, x_feas using samples
        x_cost = solution[np.recarray.argmin(solution.energy)]
        x_costs = np.asarray([compute_cost(np.asarray(sample, dtype=np.float32), self.weights) for sample in solution.sample])
        x_benefits = np.asarray([compute_benefit(np.asarray(sample, dtype=np.float32), self.benefits) for sample in solution.sample])
        x_benefits = np.where(x_costs <= self.budget, x_benefits, float('-inf'))
        x_feas = solution[np.argmax(x_benefits)]

        x_cost_weight = compute_cost(np.asarray(x_cost.sample, dtype=np.float32), self.weights)
        x_feas_weight = compute_cost(np.asarray(x_feas.sample, dtype=np.float32), self.weights)
        x_feas_benefit = np.max(x_benefits)

        return x_cost.sample, x_feas.sample, x_cost_weight, x_feas_benefit
