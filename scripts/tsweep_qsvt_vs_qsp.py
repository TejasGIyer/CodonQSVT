"""
t-Sweep: QSVT imaginary-time vs QSP unitary vs classical
==========================================================
Sweeps evolution time t and compares:
  - Classical:  pi(t) = e^{Qt} pi(0)           -- ground truth
  - QSP v1:    cos+sin channels -> e^{-iHt}    -- UNITARY, oscillatory
  - QSVT:      cosh+sinh channels -> e^{Ht}    -- DISSIPATIVE, monotonic

Also sweeps Pauli threshold to find the best accuracy-vs-depth tradeoff.

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/tsweep_qsvt_vs_qsp.py
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

from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies, ALL_SEQUENCES
from src.aae_encoding import aae_encode, get_aae_circuit
from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
from src.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
from src.block_encoding import build_simple_block_encoding
from src.qsp_circuit import (
    build_qsp_circuit, extract_codon_amps_complex, compute_full_unitary_angles,
)
from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime
from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes
from src.trotter import classical_evolution


# =====================================================================
# CONFIG
# =====================================================================
KAPPA = 1.8425
OMEGA = 0.0599
THRESHOLD = 0.1
EPSILON = 1e-3
N_QUBITS = 6

T_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
THRESHOLD_VALUES = [0.5, 0.2, 0.1, 0.05]


def dist_fidelity(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    sp, sq = float(p.sum()), float(q.sum())
    if sp > 1e-12: p = p / sp
    if sq > 1e-12: q = q / sq
    return float(np.clip((np.sqrt(p * q)).sum() ** 2, 0.0, 1.0))


# =====================================================================
# Evaluate QSP (unitary cos+sin) at one t
# =====================================================================
def evaluate_qsp_at_t(be_circuit, aae_circuit, Q, pi, sense_codons,
                      alpha, n_be, t, n_codons=61):
    phis_cos, phis_sin, _ = compute_full_unitary_angles(alpha, t, epsilon=EPSILON)
    qc_cos, info_cos = build_qsp_circuit(
        be_circuit, phis_cos, aae_circuit, N_QUBITS, n_be)
    qc_sin, info_sin = build_qsp_circuit(
        be_circuit, phis_sin, aae_circuit, N_QUBITS, n_be)
    n_total = info_cos['n_total_qubits']

    sv_cos = np.asarray(Statevector.from_instruction(qc_cos).data)
    sv_sin = np.asarray(Statevector.from_instruction(qc_sin).data)
    amps_cos = extract_codon_amps_complex(sv_cos, n_total, n_be, N_QUBITS, n_codons)
    amps_sin = extract_codon_amps_complex(sv_sin, n_total, n_be, N_QUBITS, n_codons)

    # QSP v1 combination: Re(cos)^2 + Re(sin)^2 -> normalize
    combined = amps_cos.real ** 2 + amps_sin.real ** 2
    psum = float(np.sum(combined))
    probs = combined / psum if psum > 1e-12 else np.zeros(n_codons)
    raw_norm2 = float(np.sum(np.abs(amps_cos) ** 2) + np.sum(np.abs(amps_sin) ** 2))

    pi_cl, _ = classical_evolution(Q, pi, t)
    f = dist_fidelity(pi_cl, probs)
    return f, raw_norm2, len(phis_cos) + len(phis_sin)


# =====================================================================
# Evaluate QSVT (imaginary-time cosh+sinh) at one t
# =====================================================================
def evaluate_qsvt_at_t(be_circuit, aae_circuit, Q, pi, sense_codons,
                       alpha, n_be, t, n_codons=61):
    phases_cosh, phases_sinh, ang_info = compute_qsvt_angles_imagtime(
        alpha, t, epsilon=EPSILON)
    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit, phases_cosh, aae_circuit, N_QUBITS, n_be)
    qc_sinh, info_sinh = build_qsp_circuit(
        be_circuit, phases_sinh, aae_circuit, N_QUBITS, n_be)
    n_total = info_cosh['n_total_qubits']

    sv_cosh = np.asarray(Statevector.from_instruction(qc_cosh).data)
    sv_sinh = np.asarray(Statevector.from_instruction(qc_sinh).data)
    cosh_amps = extract_codon_amps_complex(sv_cosh, n_total, n_be, N_QUBITS, n_codons)
    sinh_amps = extract_codon_amps_complex(sv_sinh, n_total, n_be, N_QUBITS, n_codons)

    # QSVT combination: Re(cosh)*norm_cosh + Re(sinh)*norm_sinh
    evolved = combine_imagtime_amplitudes(
        cosh_amps, sinh_amps,
        ang_info['norm_factor_cosh'], ang_info['norm_factor_sinh'])
    probs_raw = evolved ** 2
    raw_sum = float(np.sum(probs_raw))
    probs = probs_raw / raw_sum if raw_sum > 1e-12 else np.zeros(n_codons)

    pi_cl, _ = classical_evolution(Q, pi, t)
    f = dist_fidelity(pi_cl, probs)
    return f, raw_sum, len(phases_cosh) + len(phases_sinh)


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("  t-SWEEP: QSP (unitary) vs QSVT (imag-time) vs classical")
    print("=" * 70)

    # --- Shared setup ---
    print("\n  Building Q + H + Pauli...")
    codon_freqs = pooled_codon_frequencies()
    best_v, min_err = 50.0, float('inf')
    for test_v in np.linspace(5, 200, 391):
        err = abs(calculate_implied_omega(codon_freqs, KAPPA, test_v) - OMEGA)
        if err < min_err:
            min_err, best_v = err, test_v
    print(f"  V = {best_v:.4f} (omega err = {min_err:.6f})")

    Q, sense_codons, pi, _ = build_gy94_rate_matrix(codon_freqs, kappa=KAPPA, V=best_v)
    H, _ = symmetrize_to_hamiltonian(Q, pi, n_qubits=N_QUBITS)
    pauli_full, _ = decompose_to_pauli(H, n_qubits=N_QUBITS, threshold=1e-6)

    print("\n  Loading (or training) AAE (once)...")
    s1 = build_gapdh_register(n_qubits=N_QUBITS)
    aae_json = os.path.join(_PROJECT_DIR, 'results', 'best_aae_params_gapdh.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=6, n_trials=3, maxiter=3000)
    aae_circuit = s2['circuit']
    print(f"  AAE overlap: {s2['overlap']:.6f}")

    pi_eq = pi / pi.sum()

    # ================================================================
    # PART 1: Threshold sweep at fixed t=0.5
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 1: THRESHOLD SWEEP (t=0.5)")
    print("=" * 70)
    print(f"\n  {'Threshold':>10}  {'Terms':>6}  {'Alpha':>8}  {'Ancilla':>8}  "
          f"{'F(QSP)':>8}  {'F(QSVT)':>9}  {'QSVT_norm^2':>12}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*12}")

    for th in THRESHOLD_VALUES:
        pauli_op, n_kept = filter_pauli_op(pauli_full, th)
        alpha = float(np.sum(np.abs(pauli_op.coeffs)))
        be_circuit, alpha_be, be_info = build_simple_block_encoding(
            pauli_op, n_data_qubits=N_QUBITS)
        n_be = be_info['n_ancilla']

        try:
            f_qsp, _, _ = evaluate_qsp_at_t(
                be_circuit, aae_circuit, Q, pi, sense_codons, alpha_be, n_be, 0.5)
        except Exception:
            f_qsp = float('nan')

        try:
            f_qsvt, norm2, _ = evaluate_qsvt_at_t(
                be_circuit, aae_circuit, Q, pi, sense_codons, alpha_be, n_be, 0.5)
        except Exception:
            f_qsvt, norm2 = float('nan'), float('nan')

        print(f"  {th:>10.2f}  {n_kept:>6d}  {alpha_be:>8.4f}  {n_be:>8d}  "
              f"{f_qsp:>8.4f}  {f_qsvt:>9.4f}  {norm2:>12.4f}")

    # ================================================================
    # PART 2: t-sweep at fixed threshold
    # ================================================================
    print("\n" + "=" * 70)
    print(f"  PART 2: t-SWEEP (threshold={THRESHOLD})")
    print("=" * 70)

    pauli_op, n_kept = filter_pauli_op(pauli_full, THRESHOLD)
    alpha = float(np.sum(np.abs(pauli_op.coeffs)))
    be_circuit, alpha_be, be_info = build_simple_block_encoding(
        pauli_op, n_data_qubits=N_QUBITS)
    n_be = be_info['n_ancilla']
    print(f"  Pauli terms: {n_kept}, alpha = {alpha_be:.4f}, BE ancilla = {n_be}")

    rows = []
    print(f"\n  {'t':>6}  {'F(cl,eq)':>9}  {'F(QSP)':>8}  {'F(QSVT)':>9}  "
          f"{'QSP_norm^2':>11}  {'QSVT_norm^2':>12}  {'QSVT_top':>10}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*12}  {'-'*10}")

    for t in T_VALUES:
        pi_cl, _ = classical_evolution(Q, pi, t)
        f_cl_eq = dist_fidelity(pi_cl, pi_eq)

        t0 = time.time()
        try:
            f_qsp, qsp_norm2, qsp_phases = evaluate_qsp_at_t(
                be_circuit, aae_circuit, Q, pi, sense_codons, alpha_be, n_be, t)
        except Exception as e:
            f_qsp, qsp_norm2, qsp_phases = float('nan'), float('nan'), 0
        qsp_time = time.time() - t0

        t0 = time.time()
        try:
            f_qsvt, qsvt_norm2, qsvt_phases = evaluate_qsvt_at_t(
                be_circuit, aae_circuit, Q, pi, sense_codons, alpha_be, n_be, t)
        except Exception as e:
            f_qsvt, qsvt_norm2, qsvt_phases = float('nan'), float('nan'), 0
        qsvt_time = time.time() - t0

        # Find QSVT top codon for quick sanity check
        # (re-run is wasteful but this is diagnostic, not production)
        qsvt_top = '---'
        try:
            phases_cosh, phases_sinh, ang = compute_qsvt_angles_imagtime(alpha_be, t, epsilon=EPSILON)
            qc_cosh, ic = build_qsp_circuit(be_circuit, phases_cosh, aae_circuit, N_QUBITS, n_be)
            qc_sinh, _ = build_qsp_circuit(be_circuit, phases_sinh, aae_circuit, N_QUBITS, n_be)
            sv_c = np.asarray(Statevector.from_instruction(qc_cosh).data)
            sv_s = np.asarray(Statevector.from_instruction(qc_sinh).data)
            ac = extract_codon_amps_complex(sv_c, ic['n_total_qubits'], n_be, N_QUBITS, 61)
            as_ = extract_codon_amps_complex(sv_s, ic['n_total_qubits'], n_be, N_QUBITS, 61)
            ev = combine_imagtime_amplitudes(ac, as_, ang['norm_factor_cosh'], ang['norm_factor_sinh'])
            qsvt_top = sense_codons[int(np.argmax(ev ** 2))]
        except Exception:
            pass

        print(f"  {t:>6.2f}  {f_cl_eq:>9.4f}  {f_qsp:>8.4f}  {f_qsvt:>9.4f}  "
              f"{qsp_norm2:>11.4f}  {qsvt_norm2:>12.4f}  {qsvt_top:>10}")

        rows.append({
            't': float(t),
            'f_cl_eq': f_cl_eq,
            'f_qsp': f_qsp,
            'f_qsvt': f_qsvt,
            'qsp_norm2': qsp_norm2,
            'qsvt_norm2': qsvt_norm2,
            'qsp_phases': qsp_phases,
            'qsvt_phases': qsvt_phases,
            'qsp_time_s': qsp_time,
            'qsvt_time_s': qsvt_time,
        })

    # ================================================================
    # Save results
    # ================================================================
    results_dir = os.path.join(_PROJECT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'tsweep_qsvt_vs_qsp.json')
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'kappa': KAPPA, 'omega': OMEGA, 'V': best_v,
                'threshold': THRESHOLD, 'epsilon': EPSILON,
                'n_qubits': N_QUBITS, 'n_pauli_terms': int(n_kept),
                'alpha': float(alpha_be), 'aae_overlap': float(s2['overlap']),
            },
            'rows': rows,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # ================================================================
    # Plot
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ts = [r['t'] for r in rows]
        fqsp = [r['f_qsp'] for r in rows]
        fqsvt = [r['f_qsvt'] for r in rows]
        nqsp = [r['qsp_norm2'] for r in rows]
        nqsvt = [r['qsvt_norm2'] for r in rows]
        feq = [r['f_cl_eq'] for r in rows]

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        a1, a2 = axes

        a1.plot(ts, fqsp, 'o-', label='QSP (unitary, $e^{-iHt}$)', color='#d62728')
        a1.plot(ts, fqsvt, 's-', label='QSVT imag-time ($e^{Ht}$)', color='#2ca02c')
        a1.plot(ts, feq, ':', label='Classical vs stationary $\\pi_{eq}$', color='#999999')
        a1.set_ylabel('Bhattacharyya F vs classical $\\pi(t)$')
        a1.set_title('Fidelity vs evolution time (statevector, noiseless)')
        a1.set_ylim(0, 1.05)
        a1.grid(alpha=0.3)
        a1.legend()

        a2.plot(ts, nqsp, 'o-', label='QSP (unitary)', color='#d62728')
        a2.plot(ts, nqsvt, 's-', label='QSVT imag-time', color='#2ca02c')
        a2.set_xlabel('Evolution time t')
        a2.set_ylabel('Raw $\\|$evolved state$\\|^2$')
        a2.set_title('Mass-loss: dissipative QSVT decays, unitary QSP does not')
        a2.set_ylim(0, 1.2)
        a2.grid(alpha=0.3)
        a2.legend()

        fig.tight_layout()
        plot_path = os.path.join(results_dir, 'tsweep_qsvt_vs_qsp.png')
        fig.savefig(plot_path, dpi=150)
        print(f"  Plot saved: {plot_path}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
    except Exception as e:
        print(f"  Plotting failed: {e}")


if __name__ == "__main__":
    main()
