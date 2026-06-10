# CodonQSVT

<!-- ============ BADGES (replace USER/REPO and DOI once published) ============ -->
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/qiskit-1.x-6929C4)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-informational)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.XXXX/zenodo.XXXXXXX)

> **Quantum simulation of codon-substitution dynamics via imaginary-time QSVT.**
> CodonQSVT encodes the Goldman–Yang (GY94) codon substitution model as a Hermitian
> generator and simulates its *non-unitary, dissipative* evolution — the relaxation of a
> gene's codon-frequency distribution under purifying selection — using Quantum Singular
> Value Transformation (QSVT) on a logarithmically compact 6-qubit register.

---

## Why this exists

Classical phylogenetics simulates molecular evolution with continuous-time Markov chains.
The corresponding time-evolution operator `e^{Qt}` is **stochastic, not unitary** — codon
frequencies *relax* toward equilibrium, they don't oscillate. Standard quantum Hamiltonian
simulation builds `e^{-iHt}` (unitary) and produces the wrong, oscillatory physics.

CodonQSVT instead uses **imaginary-time QSVT** to engineer `e^{Ht}` for the symmetrized,
negative-semidefinite GY94 generator, capturing the dissipative relaxation directly. The
pipeline is a faithful proof-of-concept aimed at **future fault-tolerant hardware** — we
report statevector validation and a transparent noisy-hardware *resource estimate*, and we
do **not** claim NISQ viability.

## What's in the box

- **GY94 rate matrix** with Grantham physicochemical selection, calibrated to PAML `dN/dS`.
- **Detailed-balance symmetrization** `H = D^{1/2} Q D^{-1/2}` into a Hermitian generator.
- **Approximate Amplitude Encoding (AAE)** — a shallow, hardware-efficient brickwall ansatz
  trained classically to load the empirical codon distribution onto 6 qubits.
- **LCU block encoding** of the Pauli-decomposed `H`, with a tunable truncation threshold.
- **Imaginary-time QSVT** via parity-split `cosh`/`sinh` Chebyshev channels (phases from
  [`pyqsp`](https://github.com/ichuang/pyqsp)).
- **Validation & resource estimation** against the classical CTMC reference, including a
  noisy `FakeQuebec` transpilation study.

## Headline result

A four-threshold Pauli-truncation sweep reveals a **non-monotonic fidelity–accuracy
tradeoff**: aggressive truncation can *improve* end-to-end fidelity by suppressing the
1-norm (`α`) rescaling penalty intrinsic to LCU-based QSVT. For GAPDH the optimum sits near
threshold `τ = 0.075` (25 Pauli terms, `α ≈ 3.995`).

---

## Installation

```bash
git clone https://github.com/USER/CodonQSVT.git
cd CodonQSVT
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .          # installs the codonqsvt package + dependencies
```

Dependencies (also in `requirements.txt`): `numpy`, `scipy`, `matplotlib`, `qiskit>=1.0`,
`qiskit-aer`, `qiskit-ibm-runtime`, `pyqsp`.

Verify the install:

```bash
pytest -q          # all tests should pass on a clean clone
python scripts/smoke_test.py
```

## Quickstart

```python
import numpy as np
from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from codonqsvt.constants import KAPPA, OMEGA, calibrate_V
from codonqsvt.gy94_model import build_gy94_rate_matrix
from codonqsvt.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
from codonqsvt.block_encoding import build_simple_block_encoding
from codonqsvt.aae_encoding import get_aae_circuit
from codonqsvt.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from codonqsvt.qsvt_circuit_imagtime import run_qsvt_imagtime_experiment

freqs = pooled_codon_frequencies()
V     = calibrate_V(freqs, KAPPA, OMEGA)          # grid-search V to match PAML dN/dS
Q, sense_codons, pi, _ = build_gy94_rate_matrix(freqs, kappa=KAPPA, V=V)

H, _              = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
pauli_full, _     = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
pauli_op, n_kept  = filter_pauli_op(pauli_full, threshold=0.075)
alpha             = float(np.sum(np.abs(pauli_op.coeffs)))

s1 = build_gapdh_register(n_qubits=6)
s2 = get_aae_circuit(s1, "results/best_aae_params_gapdh.json", n_layers=8)

be_circuit, alpha, be_info = build_simple_block_encoding(pauli_op, n_data_qubits=6)
phases_cosh, phases_sinh, ang = compute_qsvt_angles_imagtime(alpha, t=0.5, epsilon=1e-3)

results = run_qsvt_imagtime_experiment(
    be_circuit, phases_cosh, phases_sinh,
    ang["norm_factor_cosh"], ang["norm_factor_sinh"],
    aae_circuit=s2["circuit"], Q=Q, pi_initial=pi, sense_codons=sense_codons,
    n_be_ancilla=be_info["n_ancilla"], t=0.5, pauli_op=pauli_op,
)
print("Hellinger fidelity (reweighted):", results["f_hell_rw"])
```

Reproduce every figure/number in the paper:

```bash
python scripts/run_full_pipeline.py     # statevector validation + truncation sweep
python scripts/paper_figures.py         # regenerate figures into results/
```

---

## The quantum techniques, briefly

| Stage | Technique | Module |
|-------|-----------|--------|
| Data loading | Approximate Amplitude Encoding (brickwall PQC, L-BFGS-B trained classically) | `aae_encoding.py` |
| Generator | Detailed-balance symmetrization `H = D^{1/2} Q D^{-1/2}` | `hamiltonian.py` |
| Input model | Pauli decomposition + threshold truncation | `hamiltonian.py` |
| Block encoding | Linear Combination of Unitaries (PREPARE·SELECT·PREPARE†) | `block_encoding.py` |
| Evolution | Imaginary-time QSVT — `cosh`/`sinh` Chebyshev channels, phases via pyqsp | `qsvt_angles_imagtime.py`, `qsvt_circuit_imagtime.py` |
| Readout | Post-selection + symmetrization reweighting `a_i = sqrt(p_i / π_eq_i)` | `qsvt_circuit_imagtime.py` |
| Baseline | Unitary QSP `e^{-iHt}` and Trotterization (for contrast) | `qsp_circuit.py`, `trotter.py` |

The key conceptual move: because `H` is negative-semidefinite, `e^{Ht}` produces exponential
decay — the quantum analogue of classical thermalization — and the evolved-state norm drops
below 1, which is the signature of a genuinely *dissipative* (non-unitary) simulation.

## Scope and honesty notes

- QSVT is a **fault-tolerant** algorithm: it has no variational feedback to absorb gate
  noise. Algorithmic validation uses **noiseless statevector** simulation; the `FakeQuebec`
  runs are a **resource-estimation exercise**, not a claim of near-term hardware viability.
- The headline near-equilibrium fidelities are dominated by the invariant stationary mode;
  the scientifically decisive regime is far-from-equilibrium initial conditions (see the
  paper's discussion and `scripts/`).

## Project layout

```
codonqsvt/   core library (models, encodings, QSVT pipeline)
data/        GAPDH coding sequences + classical register builder
scripts/     reproduce-the-paper entry points
tests/       pytest suite (block-encoding correctness, AAE, angles)
results/     small JSON artifacts + figures
docs/        methods notes
```

## Citing

If you use this code or build on the method, please cite the paper:

```bibtex
@article{iyer2025codonqsvt,
  title   = {Quantum Simulation of Codon Substitution Dynamics via Imaginary-Time QSVT},
  author  = {Iyer, Tejas Ganesh and Mishra, Sai Swapnil Kumar and Shah, Farhan Ali},
  journal = {(to appear / preprint)},
  year    = {2025},
  note    = {arXiv:XXXX.XXXXX},
  url     = {https://github.com/USER/CodonQSVT}
}
```

A machine-readable `CITATION.cff` is included so GitHub shows a **"Cite this repository"**
button automatically.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built on [Qiskit](https://www.ibm.com/quantum/qiskit) and
[pyqsp](https://github.com/ichuang/pyqsp). GY94 parameters calibrated with
[PAML](http://abacus.gene.ucl.ac.uk/software/paml.html). Grantham distances from
Grantham (1974).
