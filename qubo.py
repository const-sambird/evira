from qiskit_optimization import QuadraticProgram

def heaviside(z) -> float:
    if z > 0:
        return 1.0
    return 0.0

class GateBasedQUBO:
    def __init__(self, linear_terms: list[float], quadratic_terms: list[float]):
        '''
        A Quadratic Unconstrained Binary Optimisation (QUBO)
        expression. Represents a QUBO of the following form:

        a_1 x_1 + a_2 x_2 + ... + a_i x_i + b_11 x_1 x_1 + ... + b_1i x_1 x_i + ... + b_ii x_i x_i

        Returns an expression that may be consumed by Qiskit's
        QAOA methods.
        '''
        self.linear_coefficients = linear_terms
        self.quadratic_coefficients = quadratic_terms
        self.n_terms = len(linear_terms)
    
    def compute_state_dicts(self):
        self.linear_terms = {}
        self.quadratic_terms = {}

        for i, coeff in enumerate(self.linear_coefficients):
            self.linear_terms[f'x{i}'] = coeff
        
        quad_idx = 0

        for i in range(self.n_terms):
            for j in range(i + 1, self.n_terms):
                self.quadratic_terms[(f'x{i}', f'x{j}')] = self.quadratic_coefficients[quad_idx]
                quad_idx += 1
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        '''
        Returns the QUBO in the format that the D-Wave annealer expects.
        '''
        if (not self.linear_terms) or (not self.quadratic_terms):
            self.compute_state_dicts()
        
        self.qubo = QuadraticProgram()
        for i in range(self.n_terms):
            self.qubo.binary_var(f'x{i}')
        self.qubo.minimize(constant=0, linear=self.linear_terms, quadratic=self.quadratic_terms)

        return self.qubo.to_ising()

class AnnealerQUBO:
    def __init__(self, linear_terms: list[float], quadratic_terms: list[float]):
        '''
        A Quadratic Unconstrained Binary Optimisation (QUBO)
        expression. Represents a QUBO of the following form:

        a_1 x_1 + a_2 x_2 + ... + a_i x_i + b_11 x_1 x_1 + ... + b_1i x_1 x_i + ... + b_ii x_i x_i

        Returns an expression that may be submitted to the D-Wave
        quantum annealer.
        '''
        self.linear_coefficients = linear_terms
        self.quadratic_coefficients = quadratic_terms
        self.n_terms = len(linear_terms)

    def compute_state_dicts(self):
        self.linear_terms = {}
        self.quadratic_terms = {}

        for i, coeff in enumerate(self.linear_coefficients):
            self.linear_terms[(f'x{i}', f'x{i}')] = coeff
        
        quad_idx = 0

        for i in range(self.n_terms):
            for j in range(i + 1, self.n_terms):
                self.quadratic_terms[(f'x{i}', f'x{j}')] = self.quadratic_coefficients[quad_idx]
                quad_idx += 1
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        '''
        Returns the QUBO in the format that the D-Wave annealer expects.
        '''
        if (not self.linear_terms) or (not self.quadratic_terms):
            self.compute_state_dicts()
        
        return {**self.linear_terms, **self.quadratic_terms}

    def __repr__(self):
        if (not self.linear_terms) or (not self.quadratic_terms):
            return f'uncomputed QUBO with {self.n_terms} terms'
        
        return str(self.linear_coefficients) + 'x_i + ' + str(self.quadratic_coefficients) + 'x_i x_j'

class IndexSelectionQUBO:
    def __init__(self, benefits: list[float], weights: list[float], budget: float,
                 lam: float, rho: float, type: str = 'anneal'):
        '''
        A QUBO for database index selection.

        :param benefits: a list of the benefits we want to maximise; how beneficial
                         adding the *i*th index is to the workload execution time
        :param costs:    a list of the costs we are bounding by the storage budget;
                         how much memory it will take to materialise the *i*th index
        :param budget:   how much memory we can expend on indexes
        :param lam:      lambda - penalty term coefficient
        :param rho:      penalty term multiplier
        '''
        assert type == 'anneal' or type == 'qaoa', 'what type of QUBO do you want?'

        self.benefits = benefits
        self.weights = weights
        self.budget = budget
        self.lam = lam
        self.rho = rho
        self.n_terms = len(benefits)
        self.z = 0
        self.type = type
    
    def compute_coefficients(self):
        '''
        Compute the linear x_i and quadratic x_i x_j coefficients based on the
        full QUBO form. The derivation is found in the supporting documentation.
        '''
        self.linear_terms = [0 for _ in range(self.n_terms)]
        self.quadratic_terms = [0 for _ in range((self.n_terms * (self.n_terms - 1)) // 2)]
        quad_idx = 0

        for i in range(self.n_terms):
            vi = self.benefits[i]
            wi = self.weights[i]
            self.linear_terms[i] = (-1 * vi) + (self.lam * wi) + ((self.rho / 2) * wi * wi) \
                                   - (self.rho * self.budget * wi) - (self.rho * self.z * wi)
            for j in range(i + 1, self.n_terms):
                self.quadratic_terms[quad_idx] = self.rho * wi * self.weights[j]
                quad_idx += 1
    
    def update_classical_vals(self, lam: float = None, z: float = None):
        '''
        Update λ and z, the values in the QUBO updated by classical
        optimisation procedures, and recompute the QUBO.

        :param lam: the new λ, or None if not to be updated
        :param z:   the new z, or None if not to be updated
        '''
        if lam is None and z is None:
            return
        if lam is not None:
            self.lam = lam
        if z is not None:
            self.z = z
        self.compute_coefficients()
    
    def get_qubo(self) -> dict[tuple[str, str], float]:
        if self.type == 'anneal':
            self.qubo = AnnealerQUBO(self.linear_terms, self.quadratic_terms)
            self.qubo.compute_state_dicts()
            return self.qubo.get_qubo()
        elif self.type == 'qaoa':
            self.qubo = GateBasedQUBO(self.linear_terms, self.quadratic_terms)
            self.qubo.compute_state_dicts()
            return self.qubo.get_qubo()
        