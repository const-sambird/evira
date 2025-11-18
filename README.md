# evira

The **E**fficient **V**ariable Encoding **I**ndex **R**ecommendation **A**lgorithm is an algorithm that generates index configurations (given they are formulated as combinatorial optimisation problems) using quantum annealing.

## Operation

The example problems can be found in [`problem.py`](./common/problem.py). Each problem needs a list of benefits for each index candidate, cost of materialising that candidate, and maximum storage budget.

### Installation

Install Python (built on 3.12.0) and ensure venv is present, then create a venv and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python run.py evira {anneal,qaoa}
```

Further command line values will modify the algorithm's behaviour:

```
usage: run.py [-h] [-r QAOA_REPS] [-s QAOA_SHOTS] [--rho RHO] [-t T_MAX] [--t-conv T_CONV] [--epsilon EPSILON] [--lam LAM] [--num-partitions NUM_PARTITIONS] [-q] [-p {I5,I6,I7,CDB_I7_ADJ,CDB_I7,Trummer,I8}]
              {evira,trummer} {anneal,qaoa}

positional arguments:
  {evira,trummer}       which index selection algorithm to use
  {anneal,qaoa}         use annealing optimisation or gate-based QAOA?

options:
  -h, --help            show this help message and exit
  -r QAOA_REPS, --qaoa-reps QAOA_REPS
                        the number of the repetitions in the QAOA ansatz
  -s QAOA_SHOTS, --qaoa-shots QAOA_SHOTS
                        number of shots for the QAOA sampler
  --rho RHO             rho, penalty multiplier
  -t T_MAX, --t-max T_MAX
                        t_max, maximum number of iterations
  --t-conv T_CONV       t_conv, maximum number of iterations without improvement in x_feas
  --epsilon EPSILON     slack variable convergence criterion
  --lam LAM             trummer -- penalty term lambda
  --num-partitions NUM_PARTITIONS
                        trummer -- index candidate partitions
  -q, --quantum         run on real quantum hardware instead of simulating
  -p {I5,I6,I7,CDB_I7_ADJ,CDB_I7,Trummer}, --problem {I5,I6,I7,CDB_I7_ADJ,CDB_I7,Trummer}
                        which index selection problem should we solve?
```
