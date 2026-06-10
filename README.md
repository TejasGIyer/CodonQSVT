<!-- ============ BADGES (replace USER paths and DOI once published) ============ -->
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

- **GY94 rate matrix** with Grantham physicochemical selection, calibrated to PAML `dN/dS`
  (`κ = 1.8425`, `ω = 0.0599`, `V = 13.5`, frozen in `src/constants.py`).
- **Detailed-balance symmetrization** `H = D^{1/2} Q D^{-1/2}` into a Hermitian generator
  with a single zero eigenvalue (the stationary distribution).
- **Approximate Amplitude Encoding (AAE)** — a shallow, hardware-efficient 8-layer brickwall
  ansatz trained classically to load the empirical codon distribution onto 6 qubits, reaching
  state overlap **O = 0.9898**.
- **LCU block encoding** of the Pauli-decomposed `H`, with a tunable truncation threshold;
  block-encoding correctness (`⟨0|U_BE|0⟩ = H/α`) is unit-tested.
- **Imaginary-time QSVT** via parity-split `cosh`/`sinh` Chebyshev channels (phases from
  [`pyqsp`](https://github.com/ichuang/pyqsp)).
- **Far-from-equilibrium experiment** that isolates genuine dynamics from equilibrium
  reconstruction — the key validation that the circuit simulates *relaxation*, not just
  re-prepares the stationary state.
- **Validation & resource estimation** against the classical CTMC reference, including a
  noisy `FakeQuebec` transpilation study.

## Headline results

- **Dissipative dynamics are captured.** Under imaginary-time QSVT the evolved-state norm
  decays with `t` (from `1.00` at `t=0` to `≈0.30` at `t=0.5` for a near-equilibrium start) —
  the quantum signature of classical thermalization — whereas the unitary QSP baseline stays
  near unity. This is the qualitative distinction `e^{Ht}` vs `e^{-iHt}`.
- **The circuit tracks real dynamics, not equilibrium.** Starting *far* from equilibrium
  (a delta on the rarest observed codon, `F_H(π(0), π_eq) = 0.027`), the QSVT trajectory
  matches the classical CTMC with mean **`F_H = 0.989`** over the early window `t ∈ (0, 0.5]`,
  while the control `F_H(CTMC, π_eq) = 0.056` confirms the state is genuinely off-equilibrium
  there. High fidelity where the control is low is the evidence that the dynamics — not
  equilibrium reconstruction — are being simulated.
- **Fidelity saturates under reduced truncation; the 1-norm sets the cost.** Across thresholds
  `τ ∈ {0.20, 0.10, 0.075, 0.05}` the reweighted Hellinger fidelity rises and then saturates
  (`0.895 → 0.897 → 0.923 → 0.925`), while the Pauli 1-norm `α` grows `2.64 → 4.96 → 6.06 → 7.08`
  and the logical `cosh` circuit depth explodes from `~4.4k` to `~574k`. Beyond `τ ≈ 0.075`
  (38 terms), retaining more terms buys negligible fidelity at steeply rising resource cost —
  so `τ ≈ 0.075` is the practical operating point on a fidelity-vs-cost basis.
- At the primary operating point (threshold `τ = 0.20`) the truncated generator retains
  **7 Pauli terms** with 1-norm **`α = 2.64`** on **6 data + 3 ancilla = 9 qubits**.

---

## Installation

```bash
git clone https://github.com/TejasGIyer/CodonQSVT.git
cd CodonQSVT
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .          # or: pip install -r requirements.txt
```

Dependencies (also in `requirements.txt`): `numpy`, `scipy`, `matplotlib`, `qiskit>=1.0`,
`qiskit-aer`, `qiskit-ibm-runtime`, `pyqsp`.

Verify the install:

```bash
pytest tests/test_block_encoding.py -v     # block-encoding correctness (5/5)
python scripts/smoke_test.py
```

## Reproduce everything

One command runs the full analysis (AAE → Pauli norms → t-sweep → far-from-equilibrium →
threshold sweep → figures) in dependency order:

```bash
python scripts/run_full_pipeline.py
```

Or run stages individually:

```bash
python scripts/tsweep_qsvt_vs_qsp_hellinger.py   # QSVT vs QSP, Hellinger + norm decay
python scripts/far_from_equilibrium.py           # dynamics vs equilibrium reconstruction
python scripts/threshold_sweep.py                # four-threshold fidelity sweep (Table 9)
python scripts/pauli_truncation_norms.py         # ||H - H_tau|| vs threshold
python scripts/paper_figures.py                  # encoding-comparison figures
```

## Quickstart

```python
import numpy as np
from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from src.constants import GY94_V, PAULI_THRESHOLD_PRIMARY
from src.gy94_model import build_gy94_rate_matrix
from src.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
from src.block_encoding import build_simple_block_encoding
from src.aae_encoding import get_aae_circuit
from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from src.qsvt_circuit_imagtime import run_qsvt_imagtime_experiment

# GY94 rate matrix with the frozen, paper-calibrated parameters (kappa=1.8425, V=13.5)
freqs = pooled_codon_frequencies()
Q, sense_codons, pi, _ = build_gy94_rate_matrix(freqs)   # defaults pulled from src.constants

H, _              = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
pauli_full, _     = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
pauli_op, n_kept  = filter_pauli_op(pauli_full, threshold=PAULI_THRESHOLD_PRIMARY)  # 0.20
alpha             = float(np.sum(np.abs(pauli_op.coeffs)))

s1 = build_gapdh_register(n_qubits=6)
s2 = get_aae_circuit(s1, "results/best_aae_params_gapdh.json")   # loads cached 8-layer AAE

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

---

## The quantum techniques, briefly

| Stage | Technique | Module |
|-------|-----------|--------|
| Data loading | Approximate Amplitude Encoding (8-layer brickwall PQC, L-BFGS-B trained classically) | `src/aae_encoding.py` |
| Generator | Detailed-balance symmetrization `H = D^{1/2} Q D^{-1/2}` | `src/hamiltonian.py` |
| Input model | Pauli decomposition + threshold truncation | `src/hamiltonian.py` |
| Block encoding | Linear Combination of Unitaries (PREPARE·SELECT·PREPARE†) | `src/block_encoding.py` |
| Evolution | Imaginary-time QSVT — `cosh`/`sinh` Chebyshev channels, phases via pyqsp | `src/qsvt_angles_imagtime.py`, `src/qsvt_circuit_imagtime.py` |
| Readout | Post-selection + symmetrization reweighting `a_i = sqrt(p_i / π_eq_i)` | `src/qsvt_circuit_imagtime.py` |
| Baseline | Unitary QSP `e^{-iHt}` and Trotterization (for contrast) | `src/qsp_circuit.py`, `src/trotter.py` |

Because `H` is negative-semidefinite, `e^{Ht}` produces exponential decay — the quantum
analogue of classical thermalization — and the evolved-state norm drops below 1, the
signature of a genuinely *dissipative* (non-unitary) simulation.

## Scope and honesty notes

- QSVT is a **fault-tolerant** algorithm: it has no variational feedback to absorb gate
  noise. Algorithmic validation uses **noiseless statevector** simulation; the `FakeQuebec`
  runs are a **resource-estimation exercise**, not a claim of near-term hardware viability.
- The truncation tradeoff is one of **resource efficiency, not a fidelity paradox**: fidelity
  saturates as more Pauli terms are retained, while the 1-norm `α` — and hence `2cosh(αt)`,
  circuit depth, and post-selection cost — grows steeply. The same 1-norm penalty also bounds
  the usable evolution time: at large `t`, `2cosh(αt)` grows exponentially and the
  reconstructed-state norm is dominated by amplification of a vanishing post-selected amplitude.
- Model parameters live in one place — `src/constants.py` — calibrated to the paper. The AAE
  training uses a fixed random seed (`src/constants.AAE_RANDOM_SEED`) for reproducibility.

## Project layout

```
src/        core library (models, encodings, QSVT pipeline, constants)
data/       GAPDH coding sequences + classical register builder
scripts/    reproduce-the-paper entry points (run_full_pipeline.py orchestrates)
tests/      pytest suite (block-encoding correctness, AAE cache, QSP)
results/    small JSON artifacts + figures
```

## Citing

If you use this code or build on the method, please cite the paper:

```bibtex
@article{iyer2026codonqsvt,
  title   = {Quantum Simulation of Codon Substitution Dynamics via Imaginary-Time QSVT},
  author  = {Iyer, Tejas Ganesh and Mishra, Sai Swapnil Kumar and Shah, Farhan Ali},
  journal = {(preprint in preparation)},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX},
  url     = {https://github.com/TejasGIyer/CodonQSVT}
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