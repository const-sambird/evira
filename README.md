# evira

The **E**fficient **V**ariable Encoding **I**ndex **R**ecommendation **A**lgorithm is an algorithm that generates index configurations (given they are formulated as combinatorial optimisation problems) using quantum annealing.

## Operation

The example problems can be found in [`problem.py`](./problem.py). Each problem needs a list of benefits for each index candidate, cost of materialising that candidate, and maximum storage budget.

Modify [`evira.py`](./evira.py) to select the desired problem, then:

### Installation

Install Python (built on 3.12.0) and ensure venv is present, then create a venv and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python evira.py
```
