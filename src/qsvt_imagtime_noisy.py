"""
QSVT Imaginary-Time Noisy Backend Experiment (FakeQuebec)
==========================================================
Runs the cosh+sinh QSVT pipeline on FakeQuebec with noise model.

For shot-based noisy simulation, we can't extract amplitudes directly.
Instead we:
  1. Run cosh and sinh circuits independently with shots
  2. Post-select each on ancilla=0000
  3. Use the statevector reference to recover signed amplitudes from
     shot counts (sign recovery from SV calibration)
  4. Combine: evolved = Re(cosh_amps)*norm_cosh + Re(sinh_amps)*norm_sinh
  5. Apply reweighting: a_i = sqrt(p_i / pi_eq_i), normalize
  6. Compare with both Bhattacharyya and Hellinger fidelity

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python src/qsvt_imagtime_noisy.py
"""

import os
import sys
import time
import gc
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from src.qsp_circuit import (
    build_qsp_circuit, extract_codon_amps_complex,
    _counts_to_postselected_probs,
)
from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes
from src.trotter import classical_evolution


# =====================================================================
# FIDELITY HELPERS
# =====================================================================

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

def reweight_probs(probs, pi_eq, n_codons=61):
    """Apply sqrt(p/pi_eq) reweighting and normalize."""
    rw = np.zeros(n_codons)
    for i in range(n_codons):
        if pi_eq[i] > 1e-15 and probs[i] > 0:
            rw[i] = np.sqrt(probs[i] / pi_eq[i])
    s = float(np.sum(rw))
    return rw / s if s > 1e-12 else np.zeros(n_codons)


# =====================================================================
# NOISY EXPERIMENT
# =====================================================================

def run_qsvt_imagtime_noisy(be_circuit, phases_cosh, phases_sinh,
                            norm_factor_cosh, norm_factor_sinh,
                            aae_circuit, Q, pi_initial, sense_codons,
                            n_data_qubits=6, n_be_ancilla=4,
                            t=0.5, shots=8192, verbose=True,
                            pauli_op=None, backend_name='quebec'):
    """
    Run QSVT imaginary-time on a noisy fake backend.

    Strategy:
      - Build cosh/sinh circuits (10 qubits each, same as SV pipeline)
      - Get SV reference for sign calibration
      - Transpile for FakeQuebec
      - Run noisy shots, post-select ancilla=0000
      - Recover signed amplitudes using SV reference signs
      - Combine cosh+sinh classically
      - Apply reweighting
      - Report all fidelity metrics
    """
    if backend_name.lower() == 'quebec':
        from qiskit_ibm_runtime.fake_provider import FakeQuebec
        fake_backend = FakeQuebec()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend_label = backend_name.capitalize()
    n_codons = len(sense_codons)
    n_total = n_be_ancilla + n_data_qubits

    if verbose:
        print(f"\n  Building cosh & sinh QSVT circuits for {backend_label}...")

    qc_cosh, info_cosh = build_qsp_circuit(
        be_circuit, phases_cosh, aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    qc_sinh, info_sinh = build_qsp_circuit(
        be_circuit, phases_sinh, aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)

    # --- Ideal SV reference (for sign calibration + ceiling) ---
    if verbose:
        print(f"  Computing ideal SV reference...")
    sv_cosh = np.asarray(Statevector.from_instruction(qc_cosh).data)
    sv_sinh = np.asarray(Statevector.from_instruction(qc_sinh).data)
    cosh_amps_id = extract_codon_amps_complex(sv_cosh, n_total, n_be_ancilla, n_data_qubits, n_codons)
    sinh_amps_id = extract_codon_amps_complex(sv_sinh, n_total, n_be_ancilla, n_data_qubits, n_codons)

    # Ideal combination
    evolved_id = combine_imagtime_amplitudes(
        cosh_amps_id, sinh_amps_id, norm_factor_cosh, norm_factor_sinh)
    probs_id_raw = evolved_id ** 2
    s = float(np.sum(probs_id_raw))
    probs_id_norm = probs_id_raw / s if s > 1e-12 else np.zeros(n_codons)
    probs_id_rw = reweight_probs(probs_id_norm, pi_initial, n_codons)

    pi_classical, _ = classical_evolution(Q, pi_initial, t)

    f_ideal_bhat_raw = bhattacharyya_fidelity(pi_classical, probs_id_norm)
    f_ideal_hell_raw = hellinger_fidelity(pi_classical, probs_id_norm)
    f_ideal_bhat_rw = bhattacharyya_fidelity(pi_classical, probs_id_rw)
    f_ideal_hell_rw = hellinger_fidelity(pi_classical, probs_id_rw)

    if verbose:
        print(f"  Ideal SV ceiling:")
        print(f"    Bhat(raw)={f_ideal_bhat_raw:.4f}  Hell(raw)={f_ideal_hell_raw:.4f}")
        print(f"    Bhat(rw) ={f_ideal_bhat_rw:.4f}  Hell(rw) ={f_ideal_hell_rw:.4f}")

    # Sign references for amplitude recovery from shots
    cosh_signs = np.sign(np.real(cosh_amps_id))
    sinh_signs = np.sign(np.real(sinh_amps_id))

    # --- Transpile ---
    qc_cosh_meas = qc_cosh.copy(); qc_cosh_meas.measure_all()
    qc_sinh_meas = qc_sinh.copy(); qc_sinh_meas.measure_all()
    if verbose:
        print(f"\n  Transpiling for {backend_label} (opt level 3)...")
    t0 = time.time()
    tqc_cosh = transpile(qc_cosh_meas, backend=fake_backend, optimization_level=3)
    tqc_sinh = transpile(qc_sinh_meas, backend=fake_backend, optimization_level=3)
    transpile_time = time.time() - t0

    def _metrics(tqc):
        gc_d = dict(tqc.count_ops())
        two_q = sum(v for k, v in gc_d.items() if k in ['cx','cnot','ecr','cz','swap','iswap'])
        return {'depth': tqc.depth(), 'two_qubit_gates': two_q,
                'total_gates': sum(gc_d.values())}

    m_cosh, m_sinh = _metrics(tqc_cosh), _metrics(tqc_sinh)
    if verbose:
        print(f"  Transpiled in {transpile_time:.1f}s")
        print(f"    cosh: depth={m_cosh['depth']}, 2Q={m_cosh['two_qubit_gates']}")
        print(f"    sinh: depth={m_sinh['depth']}, 2Q={m_sinh['two_qubit_gates']}")

    # --- Free memory before noisy sim ---
    del sv_cosh, sv_sinh, qc_cosh, qc_sinh, qc_cosh_meas, qc_sinh_meas
    gc.collect()

    noise_model = NoiseModel.from_backend(fake_backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    del fake_backend; gc.collect()

    # --- Run noisy shots ---
    if verbose: print(f"\n  Running noisy cosh ({shots} shots)...")
    t0 = time.time()
    counts_cosh = noisy_sim.run(tqc_cosh, shots=shots).result().get_counts()
    cosh_time = time.time() - t0
    del tqc_cosh

    if verbose: print(f"  Running noisy sinh ({shots} shots)...")
    t0 = time.time()
    counts_sinh = noisy_sim.run(tqc_sinh, shots=shots).result().get_counts()
    sinh_time = time.time() - t0
    del tqc_sinh, noisy_sim; gc.collect()

    # --- Post-select and extract probabilities ---
    probs_cosh_noisy, kept_cosh, total_cosh = _counts_to_postselected_probs(
        counts_cosh, n_be_ancilla, n_codons)
    probs_sinh_noisy, kept_sinh, total_sinh = _counts_to_postselected_probs(
        counts_sinh, n_be_ancilla, n_codons)

    if verbose:
        ps_c = kept_cosh / total_cosh if total_cosh > 0 else 0
        ps_s = kept_sinh / total_sinh if total_sinh > 0 else 0
        print(f"\n  Post-selection:")
        print(f"    cosh: {kept_cosh}/{total_cosh} = {ps_c:.4f}")
        print(f"    sinh: {kept_sinh}/{total_sinh} = {ps_s:.4f}")

    # --- Recover signed amplitudes from shot counts ---
    # shots give |amp|^2 per codon. We take sqrt and apply SV-calibrated signs.
    cosh_amps_noisy = np.sqrt(np.clip(probs_cosh_noisy, 0, None)) * cosh_signs
    sinh_amps_noisy = np.sqrt(np.clip(probs_sinh_noisy, 0, None)) * sinh_signs

    # --- Combine imaginary-time ---
    evolved_noisy = combine_imagtime_amplitudes(
        cosh_amps_noisy, sinh_amps_noisy, norm_factor_cosh, norm_factor_sinh)
    probs_noisy_raw = evolved_noisy ** 2
    s = float(np.sum(probs_noisy_raw))
    probs_noisy_norm = probs_noisy_raw / s if s > 1e-12 else np.zeros(n_codons)
    probs_noisy_rw = reweight_probs(probs_noisy_norm, pi_initial, n_codons)

    # --- Also compute simple combined (no amplitude recovery, just average probs) ---
    probs_simple = (probs_cosh_noisy + probs_sinh_noisy) / 2.0
    s2 = float(np.sum(probs_simple))
    if s2 > 1e-12: probs_simple /= s2

    # --- Fidelity metrics ---
    f_noisy_bhat_raw = bhattacharyya_fidelity(pi_classical, probs_noisy_norm)
    f_noisy_hell_raw = hellinger_fidelity(pi_classical, probs_noisy_norm)
    f_noisy_bhat_rw = bhattacharyya_fidelity(pi_classical, probs_noisy_rw)
    f_noisy_hell_rw = hellinger_fidelity(pi_classical, probs_noisy_rw)
    f_simple_bhat = bhattacharyya_fidelity(pi_classical, probs_simple)
    f_simple_hell = hellinger_fidelity(pi_classical, probs_simple)

    tv_noisy_raw = 0.5 * float(np.sum(np.abs(pi_classical - probs_noisy_norm)))
    tv_noisy_rw = 0.5 * float(np.sum(np.abs(pi_classical - probs_noisy_rw)))

    if verbose:
        print(f"\n  Fidelity ({backend_label}) — QSVT imag-time:")
        print(f"    IDEAL CEILING (SV):")
        print(f"      Bhat(raw)={f_ideal_bhat_raw:.4f}  Hell(raw)={f_ideal_hell_raw:.4f}")
        print(f"      Bhat(rw) ={f_ideal_bhat_rw:.4f}  Hell(rw) ={f_ideal_hell_rw:.4f}")
        print(f"    NOISY (amplitude-recovered, combined):")
        print(f"      Bhat(raw)={f_noisy_bhat_raw:.4f}  Hell(raw)={f_noisy_hell_raw:.4f}")
        print(f"      Bhat(rw) ={f_noisy_bhat_rw:.4f}  Hell(rw) ={f_noisy_hell_rw:.4f}")
        print(f"    NOISY (simple avg probs, no amp recovery):")
        print(f"      Bhat={f_simple_bhat:.4f}  Hell={f_simple_hell:.4f}")
        print(f"    TV(noisy raw):  {tv_noisy_raw:.4f}")
        print(f"    TV(noisy rw):   {tv_noisy_rw:.4f}")

    return {
        'backend'              : backend_label,
        'classical_probs'      : pi_classical,
        'probs_ideal_norm'     : probs_id_norm,
        'probs_ideal_rw'       : probs_id_rw,
        'probs_noisy_norm'     : probs_noisy_norm,
        'probs_noisy_rw'       : probs_noisy_rw,
        'probs_simple'         : probs_simple,
        'f_ideal_bhat_raw'     : f_ideal_bhat_raw,
        'f_ideal_hell_raw'     : f_ideal_hell_raw,
        'f_ideal_bhat_rw'      : f_ideal_bhat_rw,
        'f_ideal_hell_rw'      : f_ideal_hell_rw,
        'f_noisy_bhat_raw'     : f_noisy_bhat_raw,
        'f_noisy_hell_raw'     : f_noisy_hell_raw,
        'f_noisy_bhat_rw'      : f_noisy_bhat_rw,
        'f_noisy_hell_rw'      : f_noisy_hell_rw,
        'f_simple_bhat'        : f_simple_bhat,
        'f_simple_hell'        : f_simple_hell,
        'tv_noisy_raw'         : tv_noisy_raw,
        'tv_noisy_rw'          : tv_noisy_rw,
        'kept_cosh': kept_cosh, 'total_cosh': total_cosh,
        'kept_sinh': kept_sinh, 'total_sinh': total_sinh,
        'metrics_cosh': m_cosh, 'metrics_sinh': m_sinh,
        'transpile_time_s'     : transpile_time,
        'cosh_run_time_s'      : cosh_time,
        'sinh_run_time_s'      : sinh_time,
        'shots'                : shots,
        'info_cosh'            : info_cosh,
        'info_sinh'            : info_sinh,
    }


def print_qsvt_noisy_report(results, sense_codons):
    backend = results['backend']
    mc, ms = results['metrics_cosh'], results['metrics_sinh']
    print("\n" + "=" * 70)
    print(f"  QSVT IMAGINARY-TIME NOISY REPORT — {backend}")
    print("=" * 70)
    print(f"\n  Transpiled circuits (10 qubits each):")
    print(f"    cosh: depth={mc['depth']}, 2Q={mc['two_qubit_gates']}")
    print(f"    sinh: depth={ms['depth']}, 2Q={ms['two_qubit_gates']}")
    ps_c = results['kept_cosh'] / results['total_cosh'] if results['total_cosh'] > 0 else 0
    ps_s = results['kept_sinh'] / results['total_sinh'] if results['total_sinh'] > 0 else 0
    print(f"\n  Post-selection:")
    print(f"    cosh: {results['kept_cosh']}/{results['total_cosh']} = {ps_c:.4f}")
    print(f"    sinh: {results['kept_sinh']}/{results['total_sinh']} = {ps_s:.4f}")
    print(f"\n  Fidelity ladder (vs classical CTMC):")
    print(f"    {'':>25}  {'Bhat':>8}  {'Helling':>8}")
    print(f"    {'':>25}  {'-'*8}  {'-'*8}")
    print(f"    {'Ideal SV (raw)':>25}  {results['f_ideal_bhat_raw']:>8.4f}  {results['f_ideal_hell_raw']:>8.4f}")
    print(f"    {'Ideal SV (reweighted)':>25}  {results['f_ideal_bhat_rw']:>8.4f}  {results['f_ideal_hell_rw']:>8.4f}")
    print(f"    {'Noisy (raw)':>25}  {results['f_noisy_bhat_raw']:>8.4f}  {results['f_noisy_hell_raw']:>8.4f}")
    print(f"    {'Noisy (reweighted)':>25}  {results['f_noisy_bhat_rw']:>8.4f}  {results['f_noisy_hell_rw']:>8.4f}  <-- main")
    print(f"    {'Simple avg (no amp rec)':>25}  {results['f_simple_bhat']:>8.4f}  {results['f_simple_hell']:>8.4f}")

    nc_bhat = results['f_ideal_bhat_rw'] - results['f_noisy_bhat_rw']
    nc_hell = results['f_ideal_hell_rw'] - results['f_noisy_hell_rw']
    print(f"\n  Noise cost (ideal - noisy, reweighted):")
    print(f"    Bhat: {nc_bhat:+.4f}   Hell: {nc_hell:+.4f}")

    pi_cl = results['classical_probs']
    pi_noisy = results['probs_noisy_rw']
    pi_ideal = results['probs_ideal_rw']
    print(f"\n  Top 10 codons (reweighted):")
    print(f"  {'Codon':>6}  {'CTMC':>9}  {'Ideal':>9}  {'Noisy':>9}  {'D_noisy':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        d = pi_noisy[idx] - pi_cl[idx]
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:9.6f}  "
              f"{pi_ideal[idx]:9.6f}  {pi_noisy[idx]:9.6f}  {d:+9.5f}")


# =====================================================================
# STANDALONE
# =====================================================================

if __name__ == "__main__":
    from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
    from src.aae_encoding import aae_encode, get_aae_circuit
    from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
    from src.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
    from src.block_encoding import build_simple_block_encoding, print_block_encoding_report
    from src.qsvt_angles_imagtime import compute_qsvt_angles_imagtime, print_qsvt_angles_report

    KAPPA = 1.8425
    OMEGA = 0.0599

    print("=" * 70)
    print("  QSVT IMAGINARY-TIME — NOISY (FakeQuebec)")
    print("=" * 70)

    print("\n  [1/5] Building Q + H + Pauli...")
    codon_freqs = pooled_codon_frequencies()
    best_v, min_err = 50.0, float('inf')
    for test_v in np.linspace(5, 200, 391):
        err = abs(calculate_implied_omega(codon_freqs, KAPPA, test_v) - OMEGA)
        if err < min_err:
            min_err, best_v = err, test_v
    Q, sense_codons, pi, _ = build_gy94_rate_matrix(codon_freqs, kappa=KAPPA, V=best_v)
    H, _ = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
    pauli_full, _ = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
    pauli_op, n_kept = filter_pauli_op(pauli_full, threshold=0.2)
    alpha = float(np.sum(np.abs(pauli_op.coeffs)))
    print(f"    Pauli terms: {n_kept}, alpha = {alpha:.4f}")

    print("\n  [2/5] Loading (or training) AAE...")
    s1 = build_gapdh_register(n_qubits=6)
    aae_json = os.path.join(_PROJECT_DIR, 'results', 'best_aae_params_gapdh.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=6, n_trials=3, maxiter=3000)
    print(f"    Overlap: {s2['overlap']:.6f}")

    print("\n  [3/5] Block encoding...")
    be_circuit, alpha, be_info = build_simple_block_encoding(pauli_op, n_data_qubits=6)
    print_block_encoding_report(be_info)

    T_EVOL = 0.5
    print(f"\n  [4/5] QSVT angles (alpha={alpha:.4f}, t={T_EVOL})...")
    phases_cosh, phases_sinh, ang_info = compute_qsvt_angles_imagtime(
        alpha, T_EVOL, epsilon=1e-3)
    print_qsvt_angles_report(phases_cosh, phases_sinh, ang_info)

    print(f"\n  [5/5] Running noisy experiment (FakeQuebec, 8192 shots)...")
    results = run_qsvt_imagtime_noisy(
        be_circuit=be_circuit,
        phases_cosh=phases_cosh, phases_sinh=phases_sinh,
        norm_factor_cosh=ang_info['norm_factor_cosh'],
        norm_factor_sinh=ang_info['norm_factor_sinh'],
        aae_circuit=s2['circuit'],
        Q=Q, pi_initial=pi, sense_codons=sense_codons,
        n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'],
        t=T_EVOL, shots=8192, verbose=True,
        pauli_op=pauli_op, backend_name='quebec')

    print_qsvt_noisy_report(results, sense_codons)
    print("\n  Done.")
