"""
threshold_sweep.py -- QSVT threshold-validity audit
====================================================
Audits the historical four truncation thresholds under the corrected pipeline
(kappa=1.8425, V=13.6853, lam_min spectral padding). For each Pauli
truncation threshold it runs the full imaginary-time QSVT pipeline at a fixed
evolution time t and reports the end-to-end Hellinger / Bhattacharyya fidelity
against the classical CTMC reference e^{Qt} pi(0).

Only strictly negative-definite truncated generators are simulated because the
global sinh-channel sign correction is invalid for a spectrum crossing zero.
Rejected thresholds and their reasons are written to the output. For valid
thresholds the script also logs the columns needed for:
  * Table 6 (block encoding): K, alpha, BE ancilla, total qubits, logical
    depth, and the EMPIRICAL post-selection probability (cosh & sinh) measured
    from the statevector -- not the 1/alpha^2 proxy.
  * Table 7 (polynomial channels): tau_r = alpha*t, 2cosh(tau_r), cosh degree
    and phase count, sinh degree and phase count, and the max approximation
    error on [-1, 0].

Near-equilibrium initial state (AAE), matching the original Table 9 setup, so
the result is directly comparable to the paper. (For the dynamics-isolating
far-from-equilibrium variant, see scripts/far_from_equilibrium.py.)

Run from project root:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --t 0.5
    python scripts/threshold_sweep.py --thresholds 0.20 0.10 0.075 0.05

Outputs:
    results/threshold_sweep.json   (per-threshold fidelities + Table 6/7 cols)
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

from qiskit.quantum_info import Statevector

from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from src.aae_encoding import get_aae_circuit
from src.gy94_model import build_gy94_rate_matrix
from src.hamiltonian import (
    symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op,
)
from src.block_encoding import build_simple_block_encoding
from src.qsp_circuit import build_qsp_circuit, extract_codon_amps_complex
from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes, assert_strictly_negative
from src.trotter import classical_evolution
from src.constants import (
    GY94_KAPPA, GY94_OMEGA, GY94_V, N_DATA_QUBITS, N_SENSE_CODONS,
    PAULI_FULL_THRESHOLD, PAULI_THRESHOLDS, T_EVOL_DEFAULT, AAE_N_LAYERS,
)


# -------------------------------------------------------------------------
# Fidelity helpers (match src/qsvt_imagtime_noisy.py)
# -------------------------------------------------------------------------
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


from src.hamiltonian import reweight_to_distribution


def reweight_probs(probs, pi_eq, n_codons=61):
    """Thin wrapper kept for call-site compatibility.

    Delegates to the single canonical implementation so the four copies of
    this function cannot drift apart again.
    """
    return reweight_to_distribution(probs, pi_eq, n_codons)


def postselection_probability(sv_data, n_be_ancilla):
    """Empirical P(|0_anc>) = sum of |amp|^2 over states whose low n_be bits
    are zero (the ancilla-low little-endian convention used throughout)."""
    mask = (1 << n_be_ancilla) - 1
    p = 0.0
    for idx in range(len(sv_data)):
        if (idx & mask) == 0:
            p += abs(sv_data[idx]) ** 2
    return float(p)


# -------------------------------------------------------------------------
# Single-threshold evaluation
# -------------------------------------------------------------------------
def run_one_threshold(pauli_full, aae_circuit, Q, pi, pi_eq, threshold, t,
                      epsilon, n_codons=N_SENSE_CODONS):
    pauli_op, n_kept = filter_pauli_op(pauli_full, threshold)
    assert_strictly_negative(pauli_op)
    from src.constants import assert_dissipative
    is_dissipative, lam_max = assert_dissipative(pauli_op)
    be_circuit, alpha, be_info = build_simple_block_encoding(
        pauli_op, n_data_qubits=N_DATA_QUBITS)
    n_be = be_info['n_ancilla']

    # QSVT angles (Table 7 columns come straight from ang_info)
    phases_cosh, phases_sinh, ang = compute_qsvt_angles_imagtime(
        alpha, t, epsilon=epsilon)

    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit, phases_cosh, aae_circuit, N_DATA_QUBITS, n_be)
    qc_sinh, info_sinh = build_qsp_circuit(
        be_circuit, phases_sinh, aae_circuit, N_DATA_QUBITS, n_be)
    n_total = info_cosh['n_total_qubits']

    n_total = info_cosh['n_total_qubits']

    # Full end-to-end QSVT circuit depth (logical, decomposed to match the
    # convention in block_encoding.py: decompose(reps=3) so composite gates
    # like the BE block and StatePreparation are expanded). This is the
    # logical gate-model depth, NOT the FakeQuebec-transpiled depth (Table 10).
    cosh_depth = int(qc_cosh.decompose(reps=3).depth())
    sinh_depth = int(qc_sinh.decompose(reps=3).depth())

    sv_cosh = np.asarray(Statevector.from_instruction(qc_cosh).data)
    sv_sinh = np.asarray(Statevector.from_instruction(qc_sinh).data)

    # Empirical post-selection probability (Table 6)
    ps_cosh = postselection_probability(sv_cosh, n_be)
    ps_sinh = postselection_probability(sv_sinh, n_be)

    cosh_amps = extract_codon_amps_complex(sv_cosh, n_total, n_be, N_DATA_QUBITS, n_codons)
    sinh_amps = extract_codon_amps_complex(sv_sinh, n_total, n_be, N_DATA_QUBITS, n_codons)

    evolved = combine_imagtime_amplitudes(
        cosh_amps, sinh_amps, ang['norm_factor_cosh'], ang['norm_factor_sinh'])

    raw_norm2 = float(np.sum(evolved ** 2))
    probs_norm = (evolved ** 2) / raw_norm2 if raw_norm2 > 1e-12 else np.zeros(n_codons)
    probs_rw = reweight_probs(probs_norm, pi_eq, n_codons)

    pi_cl, _ = classical_evolution(Q, pi, t)

    # raw (METHOD A) and reweighted (METHOD B) fidelities
    fb_raw = bhattacharyya_fidelity(pi_cl, probs_norm)
    fh_raw = hellinger_fidelity(pi_cl, probs_norm)
    fb_rw = bhattacharyya_fidelity(pi_cl, probs_rw)
    fh_rw = hellinger_fidelity(pi_cl, probs_rw)
    tv_rw = total_variation(pi_cl, probs_rw)

    return {
        'lam_max_H_tau': float(lam_max),
        'is_dissipative': bool(is_dissipative),
        # --- Table 9 (fidelity) ---
        'threshold': float(threshold),
        'K': int(n_kept),
        'alpha': float(alpha),
        'tau_r': float(ang['tau']),
        'f_bhattacharyya_raw': fb_raw,
        'f_hellinger_raw': fh_raw,
        'f_bhattacharyya_rw': fb_rw,
        'f_hellinger_rw': fh_rw,
        'tv_rw': tv_rw,
        'qsvt_norm2': raw_norm2,
        # --- Table 6 (block encoding) ---
        'be_ancilla': int(n_be),
        'total_qubits': int(be_info['n_total_qubits']),   # data + BE ancilla (paper convention)
        'be_depth': int(be_info['depth']),
        'cosh_circuit_depth': cosh_depth,                 # full QSVT cosh circuit (logical)
        'sinh_circuit_depth': sinh_depth,                 # full QSVT sinh circuit (logical)
        'postsel_prob_cosh': ps_cosh,
        'postsel_prob_sinh': ps_sinh,
        # --- Table 7 (polynomial channels) ---
        '2cosh_tau_r': float(ang['norm_factor_base']),
        'cosh_degree': int(ang['cosh_degree']),
        'cosh_phases': int(ang['n_cosh_phases']),
        'sinh_degree': int(ang['sinh_degree']),
        'sinh_phases': int(ang['n_sinh_phases']),
        'approx_error': float(ang['approx_error_on_neg_interval']),
    }


def main():
    ap = argparse.ArgumentParser(description="QSVT truncation-threshold validity audit")
    ap.add_argument('--t', type=float, default=T_EVOL_DEFAULT,
                    help=f"evolution time (default {T_EVOL_DEFAULT})")
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=list(PAULI_THRESHOLDS),
                    help="thresholds to sweep (default 0.20 0.10 0.075 0.05)")
    ap.add_argument('--epsilon', type=float, default=1e-3)
    ap.add_argument('--n-layers', type=int, default=AAE_N_LAYERS)
    args = ap.parse_args()

    print("=" * 78)
    print("  QSVT TRUNCATION-THRESHOLD VALIDITY AUDIT")
    print(f"  t = {args.t},  thresholds = {args.thresholds}")
    print("=" * 78)

    print("\n[1/3] Building Q, H, full Pauli decomposition (paper-calibrated)...")
    codon_freqs = pooled_codon_frequencies()
    Q, sense_codons, pi, _ = build_gy94_rate_matrix(
        codon_freqs, kappa=GY94_KAPPA, V=GY94_V)
    print(f"  kappa = {GY94_KAPPA}, V = {GY94_V}")
    H, h_info = symmetrize_to_hamiltonian(Q, pi, n_qubits=N_DATA_QUBITS)
    print(f"  zero eigenvalues = {h_info['n_zero_eigenvalues']} (should be 1)")
    pauli_full, _ = decompose_to_pauli(H, n_qubits=N_DATA_QUBITS,
                                       threshold=PAULI_FULL_THRESHOLD)
    pi_eq = pi / pi.sum()

    print(f"\n[2/3] Loading {args.n_layers}-layer AAE cache...")
    s1 = build_gapdh_register(n_qubits=N_DATA_QUBITS)
    aae_json = os.path.join(_PROJECT_DIR, 'results', 'best_aae_params_gapdh_probability.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=args.n_layers)
    aae_circuit = s2['circuit']
    print(f"  AAE overlap O = {float(s2['overlap']):.4f}  (n_layers = {s2['n_layers']})")

    print(f"\n[3/3] Sweeping thresholds at t = {args.t}...\n")
    header = (f"  {'thr':>6}  {'K':>3}  {'alpha':>7}  {'tau_r':>6}  "
              f"{'FH_rw':>7}  {'FB_rw':>7}  {'FH_raw':>7}  {'|QSVT|2':>8}  "
              f"{'PS_cosh':>8}  {'PS_sinh':>8}  {'2cosh':>7}  "
              f"{'cosh_d':>6}  {'sinh_d':>6}  {'cD_full':>8}  {'sD_full':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows, skipped = [], []
    for thr in sorted(args.thresholds, reverse=True):
        t0 = time.time()
        try:
            r = run_one_threshold(pauli_full, aae_circuit, Q, pi, pi_eq,
                                  thr, args.t, args.epsilon)
        except Exception as e:
            print(f"  threshold {thr}: FAILED -- {e}")
            skipped.append({'threshold': float(thr), 'reason': str(e)})
            continue
        dt = time.time() - t0
        r['eval_time_s'] = dt
        rows.append(r)
        print(f"  {thr:>6.3f}  {r['K']:>3d}  {r['alpha']:>7.4f}  {r['tau_r']:>6.3f}  "
              f"{r['f_hellinger_rw']:>7.4f}  {r['f_bhattacharyya_rw']:>7.4f}  "
              f"{r['f_hellinger_raw']:>7.4f}  {r['qsvt_norm2']:>8.4f}  "
              f"{r['postsel_prob_cosh']:>8.4f}  {r['postsel_prob_sinh']:>8.4f}  "
              f"{r['2cosh_tau_r']:>7.3f}  {r['cosh_degree']:>6d}  {r['sinh_degree']:>6d}  "
              f"{r['cosh_circuit_depth']:>8d}  {r['sinh_circuit_depth']:>8d}"
              f"  ({dt:.1f}s)")


    nondiss = [r for r in rows if not r['is_dissipative']]
    if nondiss:
        print("\n  *** WARNING: truncation broke negative-semidefiniteness at "
              f"{len(nondiss)} of {len(rows)} thresholds:")
        for r in nondiss:
            print(f"      thr={r['threshold']:.3f}  K={r['K']:>3d}  "
                  f"lam_max = {r['lam_max_H_tau']:+.4f}")
        print("  At these thresholds e^(H_tau t) AMPLIFIES along the positive")
        print("  mode, so norm decay and 'dissipative dynamics' claims do not")
        print("  hold. Either restrict the sweep to dissipative thresholds or")
        print("  state this explicitly in the manuscript.")

    # Identify the optimum
    if rows:
        best = max(rows, key=lambda r: r['f_hellinger_rw'])
        print("\n  " + "=" * 60)
        print(f"  Peak Hellinger fidelity (reweighted): FH = {best['f_hellinger_rw']:.4f}")
        print(f"    at threshold = {best['threshold']}, K = {best['K']}, "
              f"alpha = {best['alpha']:.4f}")
        monotone = all(
            rows[i]['f_hellinger_rw'] >= rows[i+1]['f_hellinger_rw']
            for i in range(len(rows) - 1))
        if len(rows) == 1:
            r = rows[0]
            print(f"  Only threshold {r['threshold']} yields a negative-definite H_tau,")
            print(f"  so no ordering across thresholds can be reported. "
                  f"F_H = {r['f_hellinger_rw']:.4f}")
            print(f"  at K = {r['K']}, alpha = {r['alpha']:.4f}.")
            print("  The finding here is the CONSTRAINT: Pauli truncation destroys the")
            print("  negative-definiteness that imaginary-time QSVT requires, leaving a")
            print("  single usable operating point. Report this, not a fidelity trend.")
        elif monotone:
            print("  -> MONOTONIC in threshold: fidelity improves toward one end; "
                  "the interior-optimum 'paradox' does NOT reproduce here.")
        else:
            print("  -> Non-trivial ordering; inspect the full curve before "
                  "claiming monotonic or non-monotonic behaviour.")
        print("  " + "=" * 60)

    out_path = os.path.join(_PROJECT_DIR, 'results', 'threshold_sweep.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'kappa': GY94_KAPPA, 'omega': GY94_OMEGA, 'V': GY94_V,
                't': args.t, 'epsilon': args.epsilon,
                'n_layers': args.n_layers, 'aae_overlap': float(s2['overlap']),
                'n_qubits': N_DATA_QUBITS,
                'zero_eigenvalues': int(h_info['n_zero_eigenvalues']),
                'readout': 'sqrt(p * pi_eq) inverse-symmetrization reweight; '
                           'near-equilibrium AAE init',
            },
            'rows': rows,
            'skipped_thresholds': skipped,
        }, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
