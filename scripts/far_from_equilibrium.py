"""
Far-From-Equilibrium QSVT Experiment
====================================
The headline near-equilibrium runs (pi(0) ~ pi_eq, AAE overlap ~ 0.988) are
dominated by the invariant zero-mode of H: ~98.8% of the state never moves
under e^{Ht}, so aggregate fidelity vs the CTMC mostly measures equilibrium
RECONSTRUCTION, not DYNAMICS. This script removes that confound.

We initialize the data register FAR from equilibrium -- by default a single
codon (a delta distribution), the maximally non-stationary input -- and track
the QSVT imaginary-time trajectory against the classical CTMC reference
e^{Qt} pi(0) across a sweep of t, as the distribution RELAXES toward pi_eq.

Why this is the decisive test
-----------------------------
For a delta start, the projection onto the zero-mode is small, so the dynamics
live in the non-stationary eigenmodes -- exactly where QSVT's cosh/sinh
construction has to do real work. If the quantum trajectory tracks the CTMC
relaxation here, the circuit is genuinely simulating dynamics. We also report
two controls:

  (i)  F_H(pi_cl(t), pi_eq)      -- how far the *classical* state still is from
       equilibrium at each t. Early on this is LOW (far from eq); it rises to
       1 as t grows. This proves the trajectory is non-trivial.
  (ii) F_H(pi_cl(t), pi(0))      -- how far the classical state has moved from
       the initial delta. This FALLS from 1 as the state relaxes.

The QSVT curve F_H(pi_cl(t), QSVT(t)) should stay high THROUGHOUT, including
the early, far-from-equilibrium times where control (i) is low -- that is the
signature we could not get from the near-equilibrium demo.

Readout
-------
The symmetrization H = D^{1/2} Q D^{-1/2} biases measured probabilities by
pi_eq; the physical distribution is recovered by a_i = sqrt(p_i / pi_eq_i)
then renormalizing (METHOD B). pi_eq is the model's stationary distribution
(known a priori), while the time-evolved distribution is what we predict, so
using pi_eq in the readout does NOT leak the answer for a non-equilibrium
start -- the initial state and the trajectory are genuinely off-equilibrium.

Run from project root:
    python scripts/far_from_equilibrium.py
    python scripts/far_from_equilibrium.py --init uniform
    python scripts/far_from_equilibrium.py --init perturbed --threshold 0.075

Outputs:
    results/far_from_equilibrium.json   (trajectory data + config)
"""

import os
import sys
import time
import json
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation

from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from src.gy94_model import build_gy94_rate_matrix
from src.hamiltonian import (
    symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op,
)
from src.block_encoding import build_simple_block_encoding
from src.qsp_circuit import build_qsp_circuit, extract_codon_amps_complex
from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes
from src.trotter import classical_evolution
from src.constants import (
    GY94_KAPPA, GY94_OMEGA, GY94_V, N_DATA_QUBITS, N_SENSE_CODONS,
    PAULI_FULL_THRESHOLD, PAULI_THRESHOLD_PRIMARY,
)

EPSILON = 1e-3
T_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Fidelity helpers (identical conventions to src/qsvt_imagtime_noisy.py)
# ---------------------------------------------------------------------------
def bhattacharyya_fidelity(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    return float(np.clip((np.sqrt(p * q)).sum() ** 2, 0.0, 1.0))


def hellinger_fidelity(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    h2 = 0.5 * float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    return float(np.clip(1.0 - h2, 0.0, 1.0))


def total_variation(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    return 0.5 * float(np.sum(np.abs(p - q)))


def reweight_probs(probs, pi_eq, n_codons=N_SENSE_CODONS):
    rw = np.zeros(n_codons)
    for i in range(n_codons):
        if pi_eq[i] > 1e-15 and probs[i] > 0:
            rw[i] = np.sqrt(probs[i] / pi_eq[i])
    s = float(np.sum(rw))
    return rw / s if s > 1e-12 else np.zeros(n_codons)


# ---------------------------------------------------------------------------
# Initial-state construction (far-from-equilibrium options)
# ---------------------------------------------------------------------------
def make_initial_distribution(kind, pi_eq, sense_codons, n_codons=N_SENSE_CODONS,
                              seed=0):
    """
    Build a (61,) probability distribution pi(0) far from pi_eq.

    kind:
      'single'    : delta on the single LEAST-frequent observed codon (the
                    most off-equilibrium physically meaningful start).
      'single_top': delta on the single MOST-frequent codon.
      'uniform'   : uniform over observed codons.
      'perturbed' : reversed-rank distribution (heavily anti-correlated with
                    pi_eq) -- strongly non-stationary but full support.
    Returns pi0 (normalized) and a human label.
    """
    observed = np.where(pi_eq > 0)[0]
    pi0 = np.zeros(n_codons)

    if kind == 'single':
        idx = observed[np.argmin(pi_eq[observed])]
        pi0[idx] = 1.0
        label = f"delta on least-frequent codon {sense_codons[idx]}"
    elif kind == 'single_top':
        idx = observed[np.argmax(pi_eq[observed])]
        pi0[idx] = 1.0
        label = f"delta on most-frequent codon {sense_codons[idx]}"
    elif kind == 'uniform':
        pi0[observed] = 1.0 / len(observed)
        label = "uniform over observed codons"
    elif kind == 'perturbed':
        # reverse the equilibrium ranking: give most mass to rare codons
        order = observed[np.argsort(pi_eq[observed])]          # ascending freq
        weights = np.linspace(1.0, 0.0, len(order)) ** 2 + 1e-3
        pi0[order] = weights
        pi0 /= pi0.sum()
        label = "rank-reversed (anti-correlated with pi_eq)"
    else:
        raise ValueError(f"unknown init kind: {kind}")

    pi0 = pi0 / pi0.sum()
    return pi0, label


def prepare_state_circuit(pi0, n_qubits=N_DATA_QUBITS):
    """
    Exact StatePreparation of |psi0> = sum_i sqrt(pi0_i) |i> on the data
    register. For a far-from-equilibrium study we use exact preparation rather
    than AAE: AAE infidelity (~0.012) would otherwise contaminate the very
    signal we are trying to isolate. The pipeline only needs *a* circuit that
    prepares the initial amplitudes on the data qubits, which this provides.
    """
    n_states = 2 ** n_qubits
    amps = np.zeros(n_states)
    amps[:len(pi0)] = np.sqrt(np.clip(pi0, 0, None))
    nrm = np.linalg.norm(amps)
    if nrm > 0:
        amps /= nrm
    qc = QuantumCircuit(n_qubits, name='prep_init')
    qc.append(StatePreparation(amps), list(range(n_qubits)))
    return qc


# ---------------------------------------------------------------------------
# QSVT imaginary-time evaluation at a single t for arbitrary initial state
# ---------------------------------------------------------------------------
def evaluate_qsvt_at_t(be_circuit, init_circuit, Q, pi0, pi_eq,
                       alpha, n_be, t, n_codons=N_SENSE_CODONS):
    if t == 0.0:
        p0 = pi0 / pi0.sum()
        return (bhattacharyya_fidelity(p0, p0), hellinger_fidelity(p0, p0),
                total_variation(p0, p0), 1.0, p0.copy(), 0)

    phases_cosh, phases_sinh, ang_info = compute_qsvt_angles_imagtime(
        alpha, t, epsilon=EPSILON)
    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit, phases_cosh, init_circuit, N_DATA_QUBITS, n_be)
    qc_sinh, _ = build_qsp_circuit(
        be_circuit, phases_sinh, init_circuit, N_DATA_QUBITS, n_be)
    n_total = info_cosh['n_total_qubits']

    sv_cosh = np.asarray(Statevector.from_instruction(qc_cosh).data)
    sv_sinh = np.asarray(Statevector.from_instruction(qc_sinh).data)
    cosh_amps = extract_codon_amps_complex(sv_cosh, n_total, n_be, N_DATA_QUBITS, n_codons)
    sinh_amps = extract_codon_amps_complex(sv_sinh, n_total, n_be, N_DATA_QUBITS, n_codons)

    evolved = combine_imagtime_amplitudes(
        cosh_amps, sinh_amps,
        ang_info['norm_factor_cosh'], ang_info['norm_factor_sinh'])

    raw_norm2 = float(np.sum(evolved ** 2))
    probs_norm = (evolved ** 2) / raw_norm2 if raw_norm2 > 1e-12 else np.zeros(n_codons)
    probs_rw = reweight_probs(probs_norm, pi_eq, n_codons)

    pi_cl, _ = classical_evolution(Q, pi0, t)
    f_b = bhattacharyya_fidelity(pi_cl, probs_rw)
    f_h = hellinger_fidelity(pi_cl, probs_rw)
    tv = total_variation(pi_cl, probs_rw)
    return f_b, f_h, tv, raw_norm2, probs_rw, len(phases_cosh) + len(phases_sinh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Far-from-equilibrium QSVT trajectory")
    ap.add_argument('--init', default='single',
                    choices=['single', 'single_top', 'uniform', 'perturbed'],
                    help="initial distribution (default: single = delta on rarest codon)")
    ap.add_argument('--threshold', type=float, default=PAULI_THRESHOLD_PRIMARY,
                    help=f"Pauli truncation threshold (default {PAULI_THRESHOLD_PRIMARY})")
    args = ap.parse_args()

    print("=" * 76)
    print("  FAR-FROM-EQUILIBRIUM QSVT TRAJECTORY")
    print(f"  init = {args.init}   threshold = {args.threshold}")
    print("=" * 76)

    # ---- Build CTMC + Hamiltonian using CENTRALIZED constants ----
    print("\n[1/4] Building Q, H, Pauli decomposition (paper-calibrated params)...")
    codon_freqs = pooled_codon_frequencies()
    Q, sense_codons, pi, q_info = build_gy94_rate_matrix(
        codon_freqs, kappa=GY94_KAPPA, V=GY94_V)
    print(f"  kappa = {GY94_KAPPA}, V = {GY94_V}  (frozen in src/constants.py)")
    H, h_info = symmetrize_to_hamiltonian(Q, pi, n_qubits=N_DATA_QUBITS)
    print(f"  zero eigenvalues = {h_info['n_zero_eigenvalues']} "
          f"(should be 1: the stationary mode)")
    print(f"  spectral window  = [{h_info['eigenvalue_min']:.4f}, "
          f"{h_info['eigenvalue_max']:.4f}]")
    pauli_full, _ = decompose_to_pauli(H, n_qubits=N_DATA_QUBITS,
                                       threshold=PAULI_FULL_THRESHOLD)

    pi_eq = pi / pi.sum()

    # ---- Far-from-equilibrium initial distribution ----
    print("\n[2/4] Constructing far-from-equilibrium initial state...")
    pi0, init_label = make_initial_distribution(args.init, pi_eq, sense_codons)
    init_circuit = prepare_state_circuit(pi0, n_qubits=N_DATA_QUBITS)
    f0_eq = hellinger_fidelity(pi0, pi_eq)
    print(f"  init: {init_label}")
    print(f"  F_H(pi(0), pi_eq) = {f0_eq:.4f}  "
          f"(LOW => genuinely far from equilibrium)")

    # ---- Block-encoding ----
    print(f"\n[3/4] Block-encoding at threshold {args.threshold}...")
    pauli_op, n_kept = filter_pauli_op(pauli_full, args.threshold)
    be_circuit, alpha_be, be_info = build_simple_block_encoding(
        pauli_op, n_data_qubits=N_DATA_QUBITS)
    n_be = be_info['n_ancilla']
    print(f"  Pauli terms = {n_kept}, alpha = {alpha_be:.4f}, BE ancilla = {n_be}")

    # ---- Trajectory sweep ----
    print("\n[4/4] Trajectory sweep (relaxation toward equilibrium)...")
    print(f"\n  {'t':>5}  {'F_cl_eq':>8}  {'F_cl_pi0':>9}  "
          f"{'F_H(QSVT)':>10}  {'F_B(QSVT)':>10}  {'TV(QSVT)':>9}  {'|QSVT|^2':>9}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}")

    rows = []
    for t in T_VALUES:
        pi_cl, _ = classical_evolution(Q, pi0, t)
        # Controls: how non-trivial is the classical trajectory at this t?
        f_cl_eq = hellinger_fidelity(pi_cl, pi_eq)    # rises 0->1 as it relaxes
        f_cl_pi0 = hellinger_fidelity(pi_cl, pi0)     # falls 1->? as it moves

        t0 = time.time()
        try:
            f_b, f_h, tv, norm2, probs_rw, nphi = evaluate_qsvt_at_t(
                be_circuit, init_circuit, Q, pi0, pi_eq, alpha_be, n_be, t)
        except Exception as e:
            print(f"   QSVT @ t={t} failed: {e}")
            f_b = f_h = tv = norm2 = float('nan'); probs_rw = None; nphi = 0
        dt = time.time() - t0

        print(f"  {t:>5.2f}  {f_cl_eq:>8.4f}  {f_cl_pi0:>9.4f}  "
              f"{f_h:>10.4f}  {f_b:>10.4f}  {tv:>9.4f}  {norm2:>9.4f}")

        rows.append({
            't': float(t),
            'f_classical_vs_eq': f_cl_eq,
            'f_classical_vs_pi0': f_cl_pi0,
            'f_hellinger_qsvt': f_h,
            'f_bhattacharyya_qsvt': f_b,
            'tv_qsvt': tv,
            'qsvt_norm2': norm2,
            'qsvt_n_phases': nphi,
            'eval_time_s': dt,
        })

    # ---- Interpretation summary ----
    early = [r for r in rows if 0.0 < r['t'] <= 0.5]
    if early:
        mean_far_fid = float(np.nanmean([r['f_hellinger_qsvt'] for r in early]))
        mean_ctrl = float(np.nanmean([r['f_classical_vs_eq'] for r in early]))
        print("\n  Interpretation (early, far-from-equilibrium window t in (0, 0.5]):")
        print(f"    mean F_H(QSVT vs CTMC)   = {mean_far_fid:.4f}")
        print(f"    mean F_H(CTMC vs pi_eq)  = {mean_ctrl:.4f}  "
              f"(low => state is genuinely off-equilibrium here)")
        print("    A high QSVT fidelity while the control is low is the evidence")
        print("    that the circuit tracks DYNAMICS, not equilibrium reconstruction.")

    # ---- Save ----
    results_dir = os.path.join(_PROJECT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'far_from_equilibrium.json')
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'init_kind': args.init,
                'init_label': init_label,
                'kappa': GY94_KAPPA,
                'omega': GY94_OMEGA,
                'V': GY94_V,
                'threshold': args.threshold,
                'epsilon': EPSILON,
                'n_qubits': N_DATA_QUBITS,
                'n_pauli_terms': int(n_kept),
                'alpha': float(alpha_be),
                'be_ancilla': int(n_be),
                'f0_init_vs_eq': f0_eq,
                'zero_eigenvalues': int(h_info['n_zero_eigenvalues']),
                'readout': 'sqrt(p/pi_eq) reweight (METHOD B); exact StatePreparation init',
            },
            'rows': rows,
        }, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()