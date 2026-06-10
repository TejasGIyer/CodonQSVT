"""
t-Sweep producing Hellinger fidelity + evolved-state-norm data
=================================================================
Generates the data needed for both target plots:

  Plot 1: Hellinger fidelity F_H vs CTMC, threshold 0.20, 8-layer AAE
          - Classical ceiling  F_H(pi(0), pi_eq)
          - QSP  e^{-iHt}  (oscillatory, unphysical)
          - QSVT e^{Ht}    (stable, reweighted)

  Plot 2: Evolved-state norm vs t
          - QSP  ||e^{-iHt}|psi>||^2 ~ sum( Re(cos)^2 + Re(sin)^2 )
            (~ 1 for ideal unitary; deviation = approximation error)
          - QSVT ||e^{Ht}|psi>||^2   ~ sum( evolved^2 )  (decays)
          - Envelope ~ e^{-2*lambda_bar*t}

Fidelity definitions (match src/qsvt_imagtime_noisy.py exactly):

    bhattacharyya_fidelity(p, q) = ( sum sqrt(p_i q_i) )^2

    hellinger_fidelity(p, q)     = 1 - (1/2) sum ( sqrt(p_i) - sqrt(q_i) )^2

    reweight(p, pi_eq)_i = sqrt(p_i / pi_eq_i)       (then normalize)

QSVT distribution = reweight(evolved^2 / sum(evolved^2), pi_eq).
This is METHOD B in the standalone tests — the one that hits ~0.9 Hellinger.

Place this file in:  C:\\Users\\Ganesh\\gene_mutation_main\\scripts\\
Run from project root:
    python scripts/tsweep_qsvt_vs_qsp_hellinger.py
"""

import os
import sys
import time
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit.quantum_info import Statevector

from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from src.aae_encoding import get_aae_circuit
from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
from src.hamiltonian import (
    symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op,
)
from src.block_encoding import build_simple_block_encoding
from src.qsp_circuit import (
    build_qsp_circuit, extract_codon_amps_complex, compute_full_unitary_angles,
)
from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes
from src.trotter import classical_evolution


# =====================================================================
# CONFIG — chosen to match the target plot titles
# =====================================================================
KAPPA       = 1.8425
OMEGA       = 0.0599
THRESHOLD   = 0.20         # matches plot title "threshold 0.20"
EPSILON     = 1e-3
N_QUBITS    = 6
N_LAYERS    = 8            # matches plot title "8-layer AAE" (cache has 8)

T_VALUES = [0.0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


# =====================================================================
# FIDELITY HELPERS — match src/qsvt_imagtime_noisy.py byte-for-byte
# =====================================================================
def bhattacharyya_fidelity(p, q):
    """F_B(p, q) = (sum sqrt(p_i q_i))^2  in [0, 1]."""
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    return float(np.clip((np.sqrt(p * q)).sum() ** 2, 0.0, 1.0))


def hellinger_fidelity(p, q):
    """F_H(p, q) = 1 - (1/2) sum ( sqrt(p_i) - sqrt(q_i) )^2  in [0, 1]."""
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    h2 = 0.5 * float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    return float(np.clip(1.0 - h2, 0.0, 1.0))


def reweight_probs(probs, pi_eq, n_codons=61):
    """
    Apply sqrt(p / pi_eq) reweighting and renormalize to a distribution.
    Identical to src/qsvt_imagtime_noisy.py::reweight_probs.
    """
    rw = np.zeros(n_codons)
    for i in range(n_codons):
        if pi_eq[i] > 1e-15 and probs[i] > 0:
            rw[i] = np.sqrt(probs[i] / pi_eq[i])
    s = float(np.sum(rw))
    return rw / s if s > 1e-12 else np.zeros(n_codons)


# =====================================================================
# QSP unitary evaluation at one t
#
# Combined construction:   probs_unnorm[i] = Re(cos)[i]^2 + Re(sin)[i]^2
# For ideal unitary e^{-iHt} this sums to ~ 1 (it IS a probability
# distribution on the post-selected branch). Deviation from 1 is the
# QSP approximation / truncation error, not unitarity loss.
#
# The "evolved state norm^2" reported for Plot 2 is therefore:
#     sum( Re(cos)^2 + Re(sin)^2 )   — one number, NOT two.
# =====================================================================
def evaluate_qsp_at_t(be_circuit, aae_circuit, Q, pi, pi_eq,
                      alpha, n_be, t, n_codons=61):
    if t == 0.0:
        pi0 = pi / pi.sum()
        f_b = bhattacharyya_fidelity(pi0, pi0)
        f_h = hellinger_fidelity(pi0, pi0)
        return f_b, f_h, 1.0, 0

    phis_cos, phis_sin, _ = compute_full_unitary_angles(alpha, t, epsilon=EPSILON)
    qc_cos, info_cos = build_qsp_circuit(
        be_circuit, phis_cos, aae_circuit, N_QUBITS, n_be)
    qc_sin, _        = build_qsp_circuit(
        be_circuit, phis_sin, aae_circuit, N_QUBITS, n_be)
    n_total = info_cos['n_total_qubits']

    sv_cos = np.asarray(Statevector.from_instruction(qc_cos).data)
    sv_sin = np.asarray(Statevector.from_instruction(qc_sin).data)
    amps_cos = extract_codon_amps_complex(sv_cos, n_total, n_be, N_QUBITS, n_codons)
    amps_sin = extract_codon_amps_complex(sv_sin, n_total, n_be, N_QUBITS, n_codons)

    combined = amps_cos.real ** 2 + amps_sin.real ** 2

    # Plot 2 quantity. pyqsp returns angles for the rescaled polynomials
    # (1/2)*cos(tau*x) and (1/2)*sin(tau*x), so Re(cos)^2 + Re(sin)^2 sums
    # to ~ 1/4 for an ideal unitary. We multiply by 4 to recover the
    # physical ||e^{-iHt}|psi>||^2, which is ~ 1 modulo approximation /
    # AAE / Pauli-truncation error.  (See header of src/qsp_circuit.py:
    # "implementing cos(alpha*t*H) up to pyqsp's 0.5 rescaling factor".)
    raw_norm2_unrescaled = float(np.sum(combined))
    raw_norm2 = 4.0 * raw_norm2_unrescaled

    probs = combined / raw_norm2_unrescaled if raw_norm2_unrescaled > 1e-12 else np.zeros(n_codons)

    pi_cl, _ = classical_evolution(Q, pi, t)
    f_b = bhattacharyya_fidelity(pi_cl, probs)
    f_h = hellinger_fidelity(pi_cl, probs)
    return f_b, f_h, raw_norm2, len(phis_cos) + len(phis_sin)


# =====================================================================
# QSVT imag-time evaluation at one t
#
# Uses the canonical METHOD B reweighting from qsvt_imagtime_noisy.py:
#
#   evolved   = (Re cosh)*N_cosh + (Re sinh)*N_sinh
#   raw_norm2 = sum(evolved^2)           — Plot 2 quantity (decays)
#   probs_norm = evolved^2 / raw_norm2   — METHOD A (raw renorm)
#   probs_rw   = sqrt(probs_norm / pi_eq), then normalize  — METHOD B
#
# METHOD B (~0.9 Hellinger) is what Plot 1 shows as the blue "stable,
# reweighted" curve.
# =====================================================================
def evaluate_qsvt_at_t(be_circuit, aae_circuit, Q, pi, pi_eq,
                       alpha, n_be, t, n_codons=61):
    if t == 0.0:
        pi0 = pi / pi.sum()
        f_b = bhattacharyya_fidelity(pi0, pi0)
        f_h = hellinger_fidelity(pi0, pi0)
        return f_b, f_h, 1.0, 0

    phases_cosh, phases_sinh, ang_info = compute_qsvt_angles_imagtime(
        alpha, t, epsilon=EPSILON)
    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit, phases_cosh, aae_circuit, N_QUBITS, n_be)
    qc_sinh, _         = build_qsp_circuit(
        be_circuit, phases_sinh, aae_circuit, N_QUBITS, n_be)
    n_total = info_cosh['n_total_qubits']

    sv_cosh = np.asarray(Statevector.from_instruction(qc_cosh).data)
    sv_sinh = np.asarray(Statevector.from_instruction(qc_sinh).data)
    cosh_amps = extract_codon_amps_complex(sv_cosh, n_total, n_be, N_QUBITS, n_codons)
    sinh_amps = extract_codon_amps_complex(sv_sinh, n_total, n_be, N_QUBITS, n_codons)

    evolved = combine_imagtime_amplitudes(
        cosh_amps, sinh_amps,
        ang_info['norm_factor_cosh'], ang_info['norm_factor_sinh'])

    # Plot 2 quantity — decays with t for dissipative QSVT
    raw_norm2 = float(np.sum(evolved ** 2))

    # Stage 1: raw normalize
    probs_raw = evolved ** 2
    probs_norm = probs_raw / raw_norm2 if raw_norm2 > 1e-12 else np.zeros(n_codons)

    # Stage 2: sqrt(p / pi_eq) reweighting — METHOD B
    probs_rw = reweight_probs(probs_norm, pi_eq, n_codons)

    pi_cl, _ = classical_evolution(Q, pi, t)
    f_b = bhattacharyya_fidelity(pi_cl, probs_rw)
    f_h = hellinger_fidelity(pi_cl, probs_rw)
    return f_b, f_h, raw_norm2, len(phases_cosh) + len(phases_sinh)


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 72)
    print("  t-SWEEP: Hellinger fidelity + evolved-state norms")
    print(f"  threshold = {THRESHOLD},  {N_LAYERS}-layer AAE")
    print("=" * 72)

    # ---------- Build CTMC + Hamiltonian ----------
    print("\n[1/4] Building Q, H, Pauli decomposition...")
    codon_freqs = pooled_codon_frequencies()
    best_v, min_err = 50.0, float('inf')
    for test_v in np.linspace(5, 200, 391):
        err = abs(calculate_implied_omega(codon_freqs, KAPPA, test_v) - OMEGA)
        if err < min_err:
            min_err, best_v = err, test_v
    print(f"  V = {best_v:.4f}  (omega err = {min_err:.6f})")

    Q, sense_codons, pi, _ = build_gy94_rate_matrix(
        codon_freqs, kappa=KAPPA, V=best_v)
    H, _ = symmetrize_to_hamiltonian(Q, pi, n_qubits=N_QUBITS)
    pauli_full, _ = decompose_to_pauli(H, n_qubits=N_QUBITS, threshold=1e-6)

    # ---------- Spectral gap for the envelope ----------
    eigvals = np.linalg.eigvalsh(H)
    abs_neg = np.abs(eigvals[eigvals < -1e-12])
    lambda_bar = float(np.mean(abs_neg)) if abs_neg.size else 0.0
    lambda_min = float(np.min(abs_neg))  if abs_neg.size else 0.0
    print(f"  spectral gap  lambda_min = {lambda_min:.4f}")
    print(f"  mean |eig|    lambda_bar = {lambda_bar:.4f}")

    # ---------- AAE (load existing 8-layer cache; overlap = 0.988) ----------
    print(f"\n[2/4] Loading cached {N_LAYERS}-layer AAE...")
    s1 = build_gapdh_register(n_qubits=N_QUBITS)
    aae_json = os.path.join(
        _PROJECT_DIR, 'results', 'best_aae_params_gapdh.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=N_LAYERS)
    aae_circuit = s2['circuit']
    overlap = float(s2['overlap'])
    print(f"  AAE overlap O = {overlap:.4f}  (n_layers = {s2['n_layers']})")

    # ---------- Block-encoding at the target threshold ----------
    print(f"\n[3/4] Block-encoding at threshold {THRESHOLD}...")
    pauli_op, n_kept = filter_pauli_op(pauli_full, THRESHOLD)
    be_circuit, alpha_be, be_info = build_simple_block_encoding(
        pauli_op, n_data_qubits=N_QUBITS)
    n_be = be_info['n_ancilla']
    print(f"  Pauli terms kept = {n_kept}")
    print(f"  alpha            = {alpha_be:.4f}")
    print(f"  BE ancilla       = {n_be}")

    # ---------- Sweep ----------
    print("\n[4/4] t-sweep...")
    pi_eq = pi / pi.sum()

    print(f"\n  {'t':>5}  {'F_cl(eq)':>9}  "
          f"{'F_H(QSP)':>9}  {'F_B(QSP)':>9}  {'|QSP|2':>8}  "
          f"{'F_H(QSVT)':>10}  {'F_B(QSVT)':>10}  {'|QSVT|2':>9}  "
          f"{'env':>8}")
    print(f"  {'-'*5}  {'-'*9}  "
          f"{'-'*9}  {'-'*9}  {'-'*8}  "
          f"{'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

    rows = []
    for t in T_VALUES:
        pi_cl, _ = classical_evolution(Q, pi, t)
        f_cl_eq = hellinger_fidelity(pi_cl, pi_eq)
        envelope = float(np.exp(-2.0 * lambda_bar * t))

        # QSP
        t0 = time.time()
        try:
            f_qsp_b, f_qsp_h, qsp_norm2, n_phi_qsp = evaluate_qsp_at_t(
                be_circuit, aae_circuit, Q, pi, pi_eq, alpha_be, n_be, t)
        except Exception as e:
            print(f"   QSP @ t={t} failed: {e}")
            f_qsp_b, f_qsp_h, qsp_norm2, n_phi_qsp = (
                float('nan'), float('nan'), float('nan'), 0)
        qsp_time = time.time() - t0

        # QSVT
        t0 = time.time()
        try:
            f_qsvt_b, f_qsvt_h, qsvt_norm2, n_phi_qsvt = evaluate_qsvt_at_t(
                be_circuit, aae_circuit, Q, pi, pi_eq, alpha_be, n_be, t)
        except Exception as e:
            print(f"   QSVT @ t={t} failed: {e}")
            f_qsvt_b, f_qsvt_h, qsvt_norm2, n_phi_qsvt = (
                float('nan'), float('nan'), float('nan'), 0)
        qsvt_time = time.time() - t0

        print(f"  {t:>5.2f}  {f_cl_eq:>9.4f}  "
              f"{f_qsp_h:>9.4f}  {f_qsp_b:>9.4f}  {qsp_norm2:>8.4f}  "
              f"{f_qsvt_h:>10.4f}  {f_qsvt_b:>10.4f}  {qsvt_norm2:>9.4f}  "
              f"{envelope:>8.4f}")

        rows.append({
            't': float(t),
            'f_classical_ceiling':  f_cl_eq,
            'f_hellinger_qsp':      f_qsp_h,
            'f_bhattacharyya_qsp':  f_qsp_b,
            'f_hellinger_qsvt':     f_qsvt_h,
            'f_bhattacharyya_qsvt': f_qsvt_b,
            'qsp_norm2':            qsp_norm2,
            'qsvt_norm2':           qsvt_norm2,
            'envelope_exp_neg_2lambda_bar_t': envelope,
            'qsp_n_phases':         n_phi_qsp,
            'qsvt_n_phases':        n_phi_qsvt,
            'qsp_time_s':           qsp_time,
            'qsvt_time_s':          qsvt_time,
        })

    # ---------- Save ----------
    results_dir = os.path.join(_PROJECT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir, 'tsweep_hellinger_and_norm.json')
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'kappa':         KAPPA,
                'omega':         OMEGA,
                'V':             best_v,
                'threshold':     THRESHOLD,
                'epsilon':       EPSILON,
                'n_qubits':      N_QUBITS,
                'n_layers':      N_LAYERS,
                'n_pauli_terms': int(n_kept),
                'alpha':         float(alpha_be),
                'aae_overlap':   overlap,
                'lambda_bar':    lambda_bar,
                'lambda_min':    lambda_min,
                'qsvt_postprocessing': 'sqrt(p/pi_eq) reweight (METHOD B)',
            },
            'rows': rows,
        }, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")
    print("\n  Next: run  python scripts/plot_hellinger_and_norm.py")


if __name__ == "__main__":
    main()
