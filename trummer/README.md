# Trummer's algorithm for database index selection

This is a Python-based implementation of the algorithm described in Trummer and Venturelli's 2024 paper, [*Leveraging Quantum Computing for Database Index Selection*](https://dl.acm.org/doi/10.1145/3665225.3665445). In particular, it implements the QUBO to apply quantum annealing/QAOA. It does not implement the minor embedding algorithm for submission to the D-Wave annealer.

This implementation was written to compare with [EVIRA](../README.md).

## Theory

The QUBO is given as $E_U$ and $E_S^{(2)}$ in Trummer's paper, but there is a change of notation involved. Let $x_i$ denote the presence of the $i$th index candidate, $v_i$ be the benefit provided by materialising the $i$th index candidate, $w_i$ be the storage cost of materialising the $i$th index candidate, $b_k^{(j)}$ be the $k$th slack binary variable in partition $j$, and $f_k$ denote the $k$th storage fraction.

The QUBO is then given by:

$$ E = -\sum_{i=1}^{|I|} v_i x_i + \lambda \cdot \Bigg ( \sum_{j=1}^{J} \Big(  \sum_{i \in I_j} w_i x_i + \sum_{k=1}^{\lceil \log W_\text{max} \rceil} f_k (b_k^{(j-1)} \cdot b_j^{(j)}) \Big)^2 \Bigg) $$

Where $\lambda >> 0$ is a penalty term.

The coefficients for the penalty portion of the QUBO are given by:

$$ (w_i^2) x_i + (f_k^2) b_k^{(j-1)} + (f_k^2) b_k^{(j)} + (2w_i f_k) x_i b_k^{(j-1)} + (2w_i f_k) x_i b_k^{(j)} + (2f_k^2) b_k^{(j-1)} b_k^{(j)} $$

Note there are three classes of binary variable in the QUBO: $x_i$, $b_k^{(j-1)}$, and $b_k^{(j)}$. Both the linear and quadratic coefficients are given above.

## Running

```
source .venv/bin/activate

python run.py [...options...] trummer {anneal,qaoa}
```