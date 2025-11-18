import numpy as np
from math import floor, log2
from qiskit_optimization import QuadraticProgram

class TrummerGateBasedQUBO:
    def __init__(self, linear_terms: list[float], quadratic_terms: list[float]):
        '''
        A Quadratic Unconstrained Binary Optimisation (QUBO)
        expression. Represents a QUBO of the following form:

        a_1 x_1 + a_2 x_2 + ... + a_i x_i + b_11 x_1 x_1 + ... + b_1i x_1 x_i + ... + b_ii x_i x_i

        Returns an expression that may be consumed by Qiskit's
        QAOA methods.
        '''
        self.linear_terms = linear_terms
        self.quadratic_terms = quadratic_terms
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        '''
        Returns the QUBO in the format that the D-Wave annealer expects.
        '''
        self.qubo = QuadraticProgram()
        for var in self.linear_terms:
            self.qubo.binary_var(var)
        self.qubo.minimize(constant=0, linear=self.linear_terms, quadratic=self.quadratic_terms)

        return self.qubo.to_ising()

class TrummerAnnealerQUBO:
    def __init__(self, linear_terms: list[float], quadratic_terms: list[float]):
        '''
        A Quadratic Unconstrained Binary Optimisation (QUBO)
        expression. Represents a QUBO of the following form:

        a_1 x_1 + a_2 x_2 + ... + a_i x_i + b_11 x_1 x_1 + ... + b_1i x_1 x_i + ... + b_ii x_i x_i

        Returns an expression that may be submitted to the D-Wave
        quantum annealer.
        '''
        self.linear_terms = linear_terms
        self.quadratic_terms = quadratic_terms
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        '''
        Returns the QUBO in the format that the D-Wave annealer expects.
        '''
        return {**self.linear_terms, **self.quadratic_terms}

class TrummerQUBO:
    def __init__(self, benefits: list[float], weights: list[float], budget: float,
                 n_space_qubits: int, lam: float, partitions: int, type: str = 'anneal'):
        '''
        A QUBO for database index selection.

        :param benefits:       a list of the benefits we want to maximise; how beneficial
                               adding the *i*th index is to the workload execution time
        :param costs:          a list of the costs we are bounding by the storage budget;
                               how much memory it will take to materialise the *i*th index
        :param budget:         how much memory we can expend on indexes
        :param n_space_qubits: how many variables will be used for slack?
        :param lam:            penalty coefficient λ
        :param partitions:     the number of partitions to divide the index candidates into
        :param type:           'anneal' or 'qaoa'
        '''
        assert type == 'anneal' or type == 'qaoa', 'what type of QUBO do you want?'

        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.lam = lam
        self.n_space_qubits = n_space_qubits
        self.n_partitions = partitions
        self.n_candidates = len(benefits)
        self.n_terms = self.n_candidates + (self.n_space_qubits * self.n_partitions)
        self.type = type

        self.f = self.compute_f()

        self.compute_coefficients()
    
    def compute_f(self):
        '''
        Returns f, an array representing the storage fraction
        coefficient values. In general, these are powers of 2,
        but the final value will be whatever is left over from
        the sum.
        '''
        f = []
        for i in range(floor(log2(self.budget))):
            f.append(2**i)
        f.append(self.budget - (2**floor(log2(self.budget))))

        assert len(f) == self.n_space_qubits, 'space qubit dimension mismatch!'

        return f
    
    def compute_coefficients(self):
        '''
        Compute the linear x_i and quadratic x_i x_j coefficients based on the
        full QUBO form. The derivation is found in the supporting documentation.
        '''
        candidates = list(range(self.n_candidates))
        partitions = np.array_split(candidates, self.n_partitions)
        self.linear_terms = [0 for _ in range(self.n_terms)]
        self.quadratic_terms = {}

        for j, p_candidates in enumerate(partitions):
            for i in p_candidates:
                vi = self.benefits[i]
                wi = self.weights[i]

                b_j_idx = self.n_candidates + (j * self.n_space_qubits)
                b_j_last_idx = self.n_candidates + ((j - 1) * self.n_space_qubits)

                self.linear_terms[i] = (-1 * vi) + (self.lam * wi * wi)
                for k in range(self.n_space_qubits):
                    self.linear_terms[b_j_idx + k] += self.lam * (self.f[k] ** 2)
                    if j > 0:
                        self.linear_terms[b_j_last_idx + k] += self.lam * (self.f[k] ** 2)

                    self.quadratic_terms[(f'x{i + 1}', f'b-{j}-{k}')] = self.lam * 2 * wi * self.f[k]
                    if j > 0:
                        self.quadratic_terms[(f'x{i + 1}', f'b-{j-1}-{k}')] = self.lam * 2 * wi * self.f[k]
                        self.quadratic_terms[(f'b-{j-1}-{k}', f'b-{j}-{k}')] = self.lam * 2 * self.f[k] * self.f[k]

        linear_dict = {}
        def linear_label(label):
            if self.type == 'anneal':
                return (label, label)
            else:
                return label
        
        for i in range(self.n_candidates):
            linear_dict[linear_label(f'x{i + 1}')] = self.linear_terms[i]
        for j in range(self.n_partitions):
            for k in range(self.n_space_qubits):
                linear_dict[linear_label(f'b-{j}-{k}')] = self.linear_terms[self.n_candidates + (j * self.n_space_qubits) + k]
        
        self.linear_terms = linear_dict
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        if self.type == 'anneal':
            self.qubo = TrummerAnnealerQUBO(self.linear_terms, self.quadratic_terms)
        elif self.type == 'qaoa':
            self.qubo = TrummerGateBasedQUBO(self.linear_terms, self.quadratic_terms)
        return self.qubo.get_qubo()
        