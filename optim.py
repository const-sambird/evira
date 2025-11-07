import numpy as np
from scipy.optimize import minimize

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.samplers import PathIntegralAnnealingSampler
from dimod import ExactSolver

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
 
#from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

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
    def __init__(self, benefits, weights, budget, reps = 1, mode = 'simulate'):
        assert mode == 'simulate' or mode == 'quantum', 'select a supported solver'
        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.reps = reps
        self.mode = mode

        #self.service = QiskitRuntimeService()
        if mode == 'quantum':
            self.backend = self.service.least_busy(
                operational=True, simulator=False, min_num_qubits=127
            )
        else:
            self.backend = GenericBackendV2(num_qubits=24)
        
        print('initialised qiskit connection with backend', self.backend)

        self.pass_manager = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
    
    def get_qaoa_circuit(self, qubo):
        qc = QAOAAnsatz(cost_operator=qubo, reps=self.reps)
        qc.measure_all()
        print(qc.parameters)

        return qc
    
    def optimise(self, qubo):
        qc = self.get_qaoa_circuit(qubo)
        candidate_circuit = self.pass_manager.run(qc)

        initial_gamma = np.pi
        initial_beta = np.pi / 2
        init_params = []

        for _ in range(self.reps):
            init_params.append(initial_beta)
        for _ in range(self.reps):
            init_params.append(initial_gamma)
        
        # from qiskit's QAOA tutorial but all this stuff is for the optimisation
        # of the trainable parameters
        objective_func_vals = []
        
        def cost_func_estimator(params, ansatz, hamiltonian, estimator):
            print('cost_func_estimator call')
            # transform the observable defined on virtual qubits to
            # an observable defined on all physical qubits
            isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        
            pub = (ansatz, isa_hamiltonian, params)
            job = estimator.run([pub])
        
            results = job.result()[0]
            cost = results.data.evs
        
            objective_func_vals.append(cost)
        
            return cost
        
        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000
        
            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
        
            result = minimize(
                cost_func_estimator,
                init_params,
                args=(candidate_circuit, qubo, estimator),
                method="COBYLA",
                tol=1e-2,
            )
            
        optimized_circuit = candidate_circuit.assign_parameters(result.x)
        sampler = Sampler(mode=self.backend)
        sampler.options.default_shots = 10000
        
        # Set simple error suppression/mitigation options
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
        
        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(100))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val / shots for key, val in counts_int.items()}
        final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}

        solution = []
        
        for bits, freq in final_distribution_int.items():
            b = list(np.binary_repr(bits, width=qc.num_qubits))
            b.reverse()
            b = [int(bit) for bit in b]
            solution.append((b, -freq))
        
        solution = np.array(solution, dtype=[('sample', 'i1', (qc.num_qubits,)), ('energy', '<f8')])
        solution = solution.view(np.recarray)
        print(solution)

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
