"""
QSVT Circuit for Imaginary-Time Evolution (e^{Ht})
====================================================
Implements e^{Ht} using cosh/sinh QSVT, for negative-semi-definite H
(the symmetrized GY94 generator).

REUSES EVERYTHING from qsp_circuit.py verbatim:
    - build_qsp_circuit         (same QSP circuit builder)
    - extract_codon_amps_complex (same amplitude extraction)
    - _build_signal_op          (same signal operator)

Only two things change:
    1. Angle inputs are cosh/sinh (from qsvt_angles_imagtime), not cos/sin
    2. Classical combination is REAL ADDITION * norm_factor, not Re^2 + Im^2

The output is a NON-UNITARY evolved state (norm <= 1 = dissipative).
Post-selection probability decays with t — this is the quantum signature
of classical thermalization.

Reference:
    Gilyen, Su, Low, Wiebe (2019), arXiv:1806.01838.
"""

import os
import sys
import time
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit.quantum_info import Statevector

# REUSE existing QSP infrastructure — no changes needed
from src.qsp_circuit import (
    build_qsp_circuit,
    extract_codon_amps_complex,
)
from src.trotter import classical_evolution

from src.hamiltonian import reweight_to_distribution
# =====================================================================
# IMAGINARY-TIME COMBINATION (the only new function)
# =====================================================================

def combine_imagtime_amplitudes(cosh_amps, sinh_amps,
                                norm_factor_cosh, norm_factor_sinh,
                                sinh_sign=-1.0):
    """
    Combine cosh and sinh channel amplitudes into the imaginary-time state.

    PARITY CORRECTION
    -----------------
    QSVT applies a definite-parity polynomial to the SINGULAR VALUES. For a
    Hermitian A with eigenvalue lam:

        even P (cosh) -> P(lam)       ... sign-insensitive, no correction
        odd  P (sinh) -> P(|lam|)     ... equals -P(lam) when lam < 0

    H_tau here is strictly negative definite (lam_max = -0.1417 at tau=0.20),
    so the sinh channel comes back globally negated and

        cosh(t*lam) + sinh(t*|lam|) = exp(-t*lam)

    i.e. the circuit evolves BACKWARDS. sinh_sign = -1 restores exp(+t*lam).

    This global sign is exact ONLY because the spectrum does not cross zero.
    Callers must verify that via assert_strictly_negative() below; on an
    indefinite H_tau the correction is per-eigenvector and this formula fails.
    """
    return (np.real(cosh_amps) * norm_factor_cosh
            + sinh_sign * np.real(sinh_amps) * norm_factor_sinh)


def assert_strictly_negative(pauli_op, tol=-1e-12, raise_on_fail=True):
    """
    Guard for the global sinh_sign correction above.

    Returns (ok, lam_max). Fails if H_tau has any eigenvalue >= 0, because
    then the odd-channel sign is eigenvector-dependent and cannot be applied
    as one global factor.
    """
    from qiskit.quantum_info import Operator
    lam_max = float(np.max(np.linalg.eigvalsh(np.real(Operator(pauli_op).data))))
    ok = lam_max < tol
    if not ok and raise_on_fail:
        raise ValueError(
            f"H_tau has lam_max = {lam_max:+.4f} >= 0. The global sinh_sign "
            f"correction in combine_imagtime_amplitudes is only valid for a "
            f"strictly negative-definite H_tau. This threshold is unusable for "
            f"imaginary-time QSVT as implemented.")
    return ok, lam_max


# =====================================================================
# FULL EXPERIMENT (statevector)
# =====================================================================

def run_qsvt_imagtime_experiment(be_circuit, phases_cosh, phases_sinh,
                                 norm_factor_cosh, norm_factor_sinh,
                                 aae_circuit, Q, pi_initial, sense_codons,
                                 n_data_qubits=6, n_be_ancilla=3,
                                 t=0.5, verbose=True, pauli_op=None,
                                 pi_eq=None):
    """
    Run the QSVT imaginary-time experiment using statevector simulation.

    Uses the SAME build_qsp_circuit as the cos/sin pipeline — only the
    angles and the combination formula differ.
    """
    if pauli_op is None:
        raise ValueError(
            "pauli_op is required to verify that the global sinh sign "
            "correction is valid for this generator")
    assert_strictly_negative(pauli_op)
    n_codons = len(sense_codons)

    # --- Build cosh circuit (reuses build_qsp_circuit) ---
    if verbose:
        print(f"\n  Building COSH QSVT circuit ({len(phases_cosh)} phases)...")
    t0 = time.time()
    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit=be_circuit, phis=phases_cosh, aae_circuit=aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    if verbose:
        print(f"  Built in {time.time()-t0:.2f}s — depth {info_cosh['depth']}, "
              f"{info_cosh['n_cx_gates']} CX, {info_cosh['n_w_applications']} walks")

    # --- Build sinh circuit (reuses build_qsp_circuit) ---
    if verbose:
        print(f"\n  Building SINH QSVT circuit ({len(phases_sinh)} phases)...")
    t0 = time.time()
    qc_sinh, info_sinh = build_qsp_circuit(
        be_circuit=be_circuit, phis=phases_sinh, aae_circuit=aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    if verbose:
        print(f"  Built in {time.time()-t0:.2f}s — depth {info_sinh['depth']}, "
              f"{info_sinh['n_cx_gates']} CX, {info_sinh['n_w_applications']} walks")

    n_total = info_cosh['n_total_qubits']

    # --- Statevector simulation, cosh channel ---
    if verbose:
        print(f"\n  Simulating COSH channel (statevector)...")
    t0 = time.time()
    sv_cosh = Statevector.from_instruction(qc_cosh)
    sv_cosh_time = time.time() - t0
    cosh_amps = extract_codon_amps_complex(
        sv_cosh.data, n_total, n_be_ancilla, n_data_qubits, n_codons)
    cosh_postsel = float(np.sum(np.abs(cosh_amps) ** 2))
    if verbose:
        print(f"  Done in {sv_cosh_time:.2f}s, post-select prob {cosh_postsel:.4f}")

    # --- Statevector simulation, sinh channel ---
    if verbose:
        print(f"  Simulating SINH channel (statevector)...")
    t0 = time.time()
    sv_sinh = Statevector.from_instruction(qc_sinh)
    sv_sinh_time = time.time() - t0
    sinh_amps = extract_codon_amps_complex(
        sv_sinh.data, n_total, n_be_ancilla, n_data_qubits, n_codons)
    sinh_postsel = float(np.sum(np.abs(sinh_amps) ** 2))
    if verbose:
        print(f"  Done in {sv_sinh_time:.2f}s, post-select prob {sinh_postsel:.4f}")

    # --- Imaginary-time combination ---
    evolved_amps_raw = combine_imagtime_amplitudes(
        cosh_amps, sinh_amps, norm_factor_cosh, norm_factor_sinh)
    evolved_probs_raw = evolved_amps_raw ** 2
    raw_sum = float(np.sum(evolved_probs_raw))

    if raw_sum > 1e-12:
        evolved_probs_normalized = evolved_probs_raw / raw_sum
    else:
        evolved_probs_normalized = np.zeros(n_codons)

    # --- Reweighted readout: exact inverse of the symmetrization ---
    #   pi_i(t) ~ sqrt(p_i * pi_eq_i)   (see src.hamiltonian docstring)
    pi_eq_used = pi_initial if pi_eq is None else pi_eq
    probs_reweighted = reweight_to_distribution(
        evolved_probs_normalized, pi_eq_used, n_codons)

    # --- Classical reference ---
    pi_classical, P_t = classical_evolution(Q, pi_initial, t)

    # --- Fidelity metrics ---
    def bhattacharyya_fidelity(p, q):
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = float(p.sum()), float(q.sum())
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        return float(np.clip((np.sqrt(p * q)).sum() ** 2, 0.0, 1.0))

    def hellinger_fidelity(p, q):
        """F_H = 1 - H^2 where H^2 = 0.5 * sum(sqrt(p) - sqrt(q))^2"""
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = float(p.sum()), float(q.sum())
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        h2 = 0.5 * float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        return float(np.clip(1.0 - h2, 0.0, 1.0))

    # Method A: raw normalized probs (no reweighting)
    f_bhat_raw = bhattacharyya_fidelity(pi_classical, evolved_probs_normalized)
    f_hell_raw = hellinger_fidelity(pi_classical, evolved_probs_normalized)

    # Inverse symmetrization: sqrt(p * pi_eq), then normalize
    f_bhat_rw = bhattacharyya_fidelity(pi_classical, probs_reweighted)
    f_hell_rw = hellinger_fidelity(pi_classical, probs_reweighted)

    tv_raw = 0.5 * float(np.sum(np.abs(pi_classical - evolved_probs_normalized)))
    tv_rw = 0.5 * float(np.sum(np.abs(pi_classical - probs_reweighted)))

    if verbose:
        print(f"\n  Mass-loss diagnostic:")
        print(f"    Raw norm^2 of evolved state: {raw_sum:.4f}")
        print(f"    (Decays with t — quantum signature of dissipation)")

        print(f"\n  Fidelity vs classical e^(Qt) pi(0):")
        print(f"    METHOD A (raw normalized probs):")
        print(f"      Bhattacharyya:  {f_bhat_raw:.6f}")
        print(f"      Hellinger:      {f_hell_raw:.6f}")
        print(f"      TV distance:    {tv_raw:.6f}")
        print(f"    inverse symmetrization (sqrt(p * pi_eq), normalized):")
        print(f"      Bhattacharyya:  {f_bhat_rw:.6f}")
        print(f"      Hellinger:      {f_hell_rw:.6f}")
        print(f"      TV distance:    {tv_rw:.6f}")

        print(f"\n  Top 5 codons (raw vs reweighted vs classical):")
        print(f"  {'Codon':>6}  {'Classical':>10}  {'Raw':>10}  {'Reweight':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
        for idx in np.argsort(pi_classical)[::-1][:5]:
            print(f"    {sense_codons[idx]:>6}  {pi_classical[idx]:10.6f}  "
                  f"{evolved_probs_normalized[idx]:10.6f}  "
                  f"{probs_reweighted[idx]:10.6f}")

    return {
        'evolved_amps_raw'         : evolved_amps_raw,
        'evolved_probs_raw'        : evolved_probs_raw,
        'evolved_probs_normalized' : evolved_probs_normalized,
        'probs_reweighted'         : probs_reweighted,
        'success_prob_raw'         : raw_sum,
        'cosh_postsel_prob'        : cosh_postsel,
        'sinh_postsel_prob'        : sinh_postsel,
        'cosh_amps'                : cosh_amps,
        'sinh_amps'                : sinh_amps,
        'classical_probs'          : pi_classical,
        'f_bhat_raw'               : f_bhat_raw,
        'f_hell_raw'               : f_hell_raw,
        'f_bhat_rw'                : f_bhat_rw,
        'f_hell_rw'                : f_hell_rw,
        'tv_raw'                   : tv_raw,
        'tv_rw'                    : tv_rw,
        'info_cosh'                : info_cosh,
        'info_sinh'                : info_sinh,
        'n_be_ancilla'             : int(n_be_ancilla),
        'sv_cosh_time'             : sv_cosh_time,
        'sv_sinh_time'             : sv_sinh_time,
    }


# =====================================================================
# REPORT
# =====================================================================

def print_qsvt_imagtime_report(results, sense_codons):
    ic = results['info_cosh']
    is_ = results['info_sinh']
    print("\n" + "=" * 70)
    print("  QSVT IMAGINARY-TIME EVOLUTION  --  REPORT")
    print("=" * 70)
    print(f"\n  Circuit metrics:")
    print(f"    Cosh: {ic['N_angles']} phases, depth {ic['depth']}, CX {ic['n_cx_gates']}")
    print(f"    Sinh: {is_['N_angles']} phases, depth {is_['depth']}, CX {is_['n_cx_gates']}")
    print(f"    Total qubits: {ic['n_total_qubits']}")
    print(f"\n  Post-selection:")
    print(f"    Cosh: {100*results['cosh_postsel_prob']:.2f}%")
    print(f"    Sinh: {100*results['sinh_postsel_prob']:.2f}%")
    print(f"    Raw evolved-state norm^2: {results['success_prob_raw']:.4f}")
    print(f"\n  Fidelity vs classical e^(Qt) pi(0):")
    print(f"    METHOD A (raw normalized):")
    print(f"      Bhattacharyya = {results['f_bhat_raw']:.6f}")
    print(f"      Hellinger     = {results['f_hell_raw']:.6f}")
    print(f"      TV distance   = {results['tv_raw']:.6f}")
    print(f"    inverse symmetrization (sqrt(p * pi_eq), normalized):")
    print(f"      Bhattacharyya = {results['f_bhat_rw']:.6f}")
    print(f"      Hellinger     = {results['f_hell_rw']:.6f}")
    print(f"      TV distance   = {results['tv_rw']:.6f}")

    pi_cl = results['classical_probs']
    pi_raw = results['evolved_probs_normalized']
    pi_rw = results['probs_reweighted']
    print(f"\n  Top 10 codons (classical vs raw vs reweighted):")
    print(f"  {'Codon':>6}  {'Classical':>10}  {'Raw':>10}  {'Reweight':>10}  {'D_raw':>8}  {'D_rw':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        d_raw = pi_raw[idx] - pi_cl[idx]
        d_rw = pi_rw[idx] - pi_cl[idx]
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:10.6f}  "
              f"{pi_raw[idx]:10.6f}  {pi_rw[idx]:10.6f}  "
              f"{d_raw:+8.5f}  {d_rw:+8.5f}")


# =====================================================================
# STANDALONE
# =====================================================================

if __name__ == "__main__":
    from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies, ALL_SEQUENCES
    from src.aae_encoding import aae_encode, get_aae_circuit
    from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
    from src.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
    from src.block_encoding import build_simple_block_encoding, print_block_encoding_report
    from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime, print_qsvt_angles_report

    

    print("=" * 70)
    print("  QSVT IMAGINARY-TIME EVOLUTION  --  STANDALONE TEST")
    print("=" * 70)

    # [1] Build Q + H + Pauli (same as QSP pipeline)
    print("\n  [1/4] Building Q + Hamiltonian + Pauli decomposition...")
    codon_freqs = pooled_codon_frequencies()
    from src.constants import GY94_KAPPA, GY94_OMEGA, GY94_V, AAE_N_LAYERS
    KAPPA, OMEGA = GY94_KAPPA, GY94_OMEGA
    codon_freqs = pooled_codon_frequencies()
    Q, sense_codons, pi, q_info = build_gy94_rate_matrix(
        codon_freqs, kappa=KAPPA, V=GY94_V)
    H, h_info = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
    pauli_full, _ = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
    pauli_op, n_kept = filter_pauli_op(pauli_full, threshold=0.20)
    alpha = float(np.sum(np.abs(pauli_op.coeffs)))
    print(f"    Pauli terms: {n_kept}, alpha = {alpha:.4f}")

    # [2] Load (or train) AAE — cached params reused across runs
    print("\n  [2/4] Loading (or training) AAE...")
    s1 = build_gapdh_register(n_qubits=6)
    aae_json = os.path.join(_PROJECT_DIR, 'results', 'best_aae_params_gapdh_probability.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=AAE_N_LAYERS, n_trials=3, maxiter=3000)
    print(f"    Overlap: {s2['overlap']:.6f}")

    # [3] Block encoding (same as QSP pipeline)
    print("\n  [3/4] Block encoding...")
    be_circuit, alpha, be_info = build_simple_block_encoding(pauli_op, n_data_qubits=6)
    print_block_encoding_report(be_info)

    # [4] QSVT angles + experiment (NEW: cosh/sinh instead of cos/sin)
    T_EVOL = 0.5
    print(f"\n  [4/4] Computing QSVT imaginary-time phases (alpha={alpha:.4f}, t={T_EVOL})...")
    phases_cosh, phases_sinh, ang_info = compute_qsvt_angles_imagtime(
        alpha, T_EVOL, epsilon=1e-3)
    print_qsvt_angles_report(phases_cosh, phases_sinh, ang_info)

    print(f"\n  Running QSVT imaginary-time experiment...")
    results = run_qsvt_imagtime_experiment(
        be_circuit=be_circuit,
        phases_cosh=phases_cosh, phases_sinh=phases_sinh,
        norm_factor_cosh=ang_info['norm_factor_cosh'],
        norm_factor_sinh=ang_info['norm_factor_sinh'],
        aae_circuit=s2['circuit'],
        Q=Q, pi_initial=pi, sense_codons=sense_codons,
        n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'],
        t=T_EVOL, verbose=True, pauli_op=pauli_op)

    print_qsvt_imagtime_report(results, sense_codons)
    print("\n  Done.")
