# Shifted QSVT validation

This folder is an isolated validation of scalar-shifted imaginary-time QSVT.
It does not modify or replace any production or hardware experiment code.

The experiment keeps more Pauli terms than the seven-term primary model,
then shifts the truncated Hamiltonian by a scalar identity term,

\[
H'_{\tau}=H_{\tau}-sI,\qquad s>\lambda_{\max}(H_{\tau}),
\]

so that its spectrum is strictly negative and the existing global odd-channel
sign correction is valid. Since

\[
e^{tH'_{\tau}}=e^{-st}e^{tH_{\tau}},
\]

the scalar shift changes the postselection norm but cancels from the normalized
codon distribution.

## Run

From the project root, using the same environment as the main pipeline:

```powershell
python shifted_qsvt_validation/run_validation.py
```

The default run:

- scans coefficient cutoffs `0.20`, `0.10`, `0.075`, and `0.05`;
- synthesizes high-precision QSP phases and runs their reconstructed ideal-QSVT
  response for the selected `0.075` cutoff;
- compares it with the exact shifted exponential, exact unshifted truncated
  exponential, the full classical CTMC, and the identity baseline;
- verifies strict negativity, shift invariance, polynomial accuracy, and
  trajectory fidelity;
- writes JSON, CSV, a text summary, and figures under `results/` in this folder.

The ideal-polynomial evaluation reconstructs the response of the actual QSP
phase sequences and applies it spectrally. Phase capitalization is tightened
from pyqsp's approximately `1e-4` default to `1e-14`, because the default
perturbation would be amplified by the long-time cosh/sinh normalization. The
run fails unless the synthesized response agrees with the target exponential.
It avoids simulating the very deep 13-qubit gate expansion and is the
appropriate FTQC-era algorithm validation.
The existing block-encoding and QSP gate-layout tests remain responsible for
the circuit-construction layer.

An optional direct gate-statevector checkpoint is available but deliberately
not part of the default run because the 38-term controlled-Pauli expansion is
extremely slow on a classical statevector simulator:

```powershell
python shifted_qsvt_validation/run_validation.py --gate-checkpoint 0.5
```

## Interpretation boundary

High fidelity here demonstrates that shifted QSVT can reproduce the normalized
CTMC trajectory for the retained operator at a substantially more accurate
Pauli cutoff. It is a proof-of-concept for the algorithmic methodology, not a
claim of quantum speedup or near-term execution of the full QSVT circuit.
