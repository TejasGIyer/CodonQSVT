"""
Step 7: Run the Full Experiment
================================
Assembles the complete quantum pipeline and runs it on two backends:

    |0...0> --[AAE]--> |psi_0> --[Trotter]--> |psi(t)> --[Measure]

Backend 1: Aer (ideal simulator)      -> pure quantum result, zero noise
Backend 2: FakeQuebec (noisy)         -> realistic hardware noise model

For each backend we collect:
    - Shot counts (measurement outcomes)
    - Density matrix (for fidelity computation)
    - Evolved codon probability distribution

Step 8 (verification) then compares both quantum results to the
exact classical answer from scipy.linalg.expm(Q * t).
"""

import os
import sys

# Add project root to path so this file works whether run directly or imported
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np
import time

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeQuebec
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity

from src.trotter import (
    build_trotter_circuit,
    build_full_evolution_circuit,
    classical_evolution,
    sweep_trotter_steps,
    print_trotter_report,
)


# =========================================================================
# CIRCUIT METRICS HELPER
# =========================================================================

def get_circuit_metrics(transpiled_circuit):
    """Extract gate counts and depth from a transpiled circuit."""
    gc = dict(transpiled_circuit.count_ops())
    two_q = sum(v for k, v in gc.items()
                if k in ['cx', 'cnot', 'ecr', 'cz', 'swap', 'iswap'])
    return {
        'depth'           : transpiled_circuit.depth(),
        'total_gates'     : sum(gc.values()),
        'gate_counts'     : gc,
        'two_qubit_gates' : two_q,
        'swap_count'      : gc.get('swap', 0),
    }


# =========================================================================
# QUANTUM PROBABILITY EXTRACTION
# =========================================================================

def counts_to_codon_probs(counts, sense_codons, n_qubits=6, shots=8192):
    """
    Convert raw measurement counts -> codon probability distribution.

    The quantum state lives in a 64-dimensional Hilbert space (6 qubits),
    but only the first 61 basis states correspond to real codons.
    We map each measurement bitstring back to a codon index.

    Parameters
    ----------
    counts      : dict  {bitstring: count}
    sense_codons: list of 61 codon strings (ordered, from gy94_model)
    n_qubits    : int
    shots       : int

    Returns
    -------
    probs : np.ndarray (61,)
    """
    n_codons = len(sense_codons)
    probs = np.zeros(n_codons)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        idx = int(bitstring[::-1], 2)
        if idx < n_codons:
            probs[idx] += count / total
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def statevector_to_codon_probs(sv, n_codons=61):
    """Extract codon probabilities from a statevector."""
    amps = np.array(sv) if not isinstance(sv, np.ndarray) else sv
    probs = np.abs(amps[:n_codons]) ** 2
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


# =========================================================================
# DENSITY MATRIX FROM COUNTS (fallback)
# =========================================================================

def _dm_from_counts(counts, num_qubits):
    """Build a diagonal density matrix from shot counts (classical mixture)."""
    n = 2 ** num_qubits
    arr = np.zeros((n, n), dtype=complex)
    total = sum(counts.values())
    for bs, c in counts.items():
        idx = int(bs[::-1], 2)
        if idx < n:
            arr[idx, idx] = c / total
    return DensityMatrix(arr)


# =========================================================================
# CORE EXPERIMENT RUNNER
# =========================================================================

def run_evolution_experiment(
    aae_circuit,
    pauli_op,
    Q,
    pi_initial,
    sense_codons,
    t=0.5,
    n_trotter_steps=3,
    trotter_order=1,
    shots=8192,
    verbose=True,
):
    """
    Run the full Step 7 experiment: build circuit, simulate on Aer and
    FakeQuebec, extract evolved codon distributions.

    Parameters
    ----------
    aae_circuit     : QuantumCircuit  (trained AAE from Step 2)
    pauli_op        : SparsePauliOp   (Hamiltonian from Step 5)
    Q               : np.ndarray (61,61)  GY94 rate matrix (Step 3)
    pi_initial      : np.ndarray (61,)    initial codon frequencies
    sense_codons    : list of str         ordered codon list
    t               : float               evolution time
    n_trotter_steps : int                 Trotter repetitions
    trotter_order   : int                 1 or 2
    shots           : int                 measurement shots per backend
    verbose         : bool

    Returns
    -------
    results : dict with keys aer, quebec, classical, circuit_info, trotter_info
    """
    n_qubits = aae_circuit.num_qubits

    # --- Build full circuit ---
    if verbose:
        print("\n  Building full evolution circuit (AAE + Trotter)...")
    t0 = time.time()
    full_circuit, full_circuit_meas, trotter_info = build_full_evolution_circuit(
        aae_circuit, pauli_op, t,
        n_trotter_steps=n_trotter_steps,
        order=trotter_order,
    )
    build_time = time.time() - t0
    if verbose:
        print(f"  Circuit built in {build_time:.2f}s")
        print_trotter_report(trotter_info, has_full_circuit=True)

    # --- Backends ---
    fake_backend = FakeQuebec()
    aer_sim      = AerSimulator()

    if verbose:
        print("\n  Transpiling for FakeQuebec (optimization_level=3)...")
    t0 = time.time()
    transpiled_quebec = transpile(full_circuit_meas, backend=fake_backend, optimization_level=3)
    transpile_time        = time.time() - t0
    quebec_metrics    = get_circuit_metrics(transpiled_quebec)

    transpiled_aer = transpile(full_circuit_meas, backend=aer_sim, optimization_level=1)
    aer_metrics    = get_circuit_metrics(transpiled_aer)

    if verbose:
        print(f"  Transpiled in {transpile_time:.2f}s")
        print(f"\n  Transpiled circuit stats:")
        print(f"    Aer:         depth={aer_metrics['depth']:5d}  gates={aer_metrics['total_gates']:6d}  2Q={aer_metrics['two_qubit_gates']:5d}")
        print(f"    Quebec:      depth={quebec_metrics['depth']:5d}  gates={quebec_metrics['total_gates']:6d}  2Q={quebec_metrics['two_qubit_gates']:5d}  SWAPs={quebec_metrics['swap_count']:4d}")

    # --- Aer simulation ---
    if verbose:
        print(f"\n  Running Aer (ideal) simulation ({shots} shots)...")
    t0 = time.time()
    aer_counts    = aer_sim.run(transpiled_aer, shots=shots).result().get_counts()
    aer_shot_time = time.time() - t0

    try:
        aer_sv       = Statevector.from_instruction(full_circuit)
        aer_dm       = DensityMatrix(aer_sv)
        aer_probs_sv = statevector_to_codon_probs(aer_sv.data, n_codons=len(sense_codons))
    except Exception:
        aer_dm       = _dm_from_counts(aer_counts, n_qubits)
        aer_probs_sv = None

    aer_probs_counts = counts_to_codon_probs(aer_counts, sense_codons, n_qubits, shots)

    if verbose:
        print(f"  Aer done in {aer_shot_time:.2f}s")
        top5 = np.argsort(aer_probs_counts)[::-1][:5]
        print(f"  Top 5 codons (Aer):")
        for idx in top5:
            print(f"    {sense_codons[idx]}: {aer_probs_counts[idx]:.5f}")

    # --- FakeQuebec simulation ---
    if verbose:
        print(f"\n  Running FakeQuebec (noisy) simulation ({shots} shots)...")
    t0 = time.time()
    quebec_counts    = fake_backend.run(transpiled_quebec, shots=shots).result().get_counts()
    quebec_shot_time = time.time() - t0

    try:
        noise_model   = NoiseModel.from_backend(fake_backend)
        dm_sim        = AerSimulator(method='density_matrix', noise_model=noise_model)
        qc_dm         = full_circuit.copy()
        qc_dm.save_density_matrix()
        dm_data       = dm_sim.run(transpile(qc_dm, backend=dm_sim, optimization_level=3)).result().data()['density_matrix']
        quebec_dm = DensityMatrix(dm_data)
    except Exception:
        quebec_dm = _dm_from_counts(quebec_counts, n_qubits)

    quebec_probs = counts_to_codon_probs(quebec_counts, sense_codons, n_qubits, shots)

    if verbose:
        print(f"  Quebec done in {quebec_shot_time:.2f}s")
        top5 = np.argsort(quebec_probs)[::-1][:5]
        print(f"  Top 5 codons (Quebec):")
        for idx in top5:
            print(f"    {sense_codons[idx]}: {quebec_probs[idx]:.5f}")

    # --- Classical reference ---
    if verbose:
        print(f"\n  Computing classical reference (scipy.linalg.expm)...")
    t0 = time.time()
    pi_classical, P_t = classical_evolution(Q, pi_initial, t)
    classical_time    = time.time() - t0

    if verbose:
        print(f"  Classical done in {classical_time*1000:.2f}ms")
        top5 = np.argsort(pi_classical)[::-1][:5]
        print(f"  Top 5 codons (classical exact):")
        for idx in top5:
            print(f"    {sense_codons[idx]}: {pi_classical[idx]:.5f}")

    circuit_info = {
        'n_qubits': n_qubits, 't': t,
        'n_trotter_steps': n_trotter_steps, 'trotter_order': trotter_order,
        'shots': shots, 'build_time_s': build_time,
        'transpile_time_s': transpile_time, 'aer_run_time_s': aer_shot_time,
        'quebec_run_time_s': quebec_shot_time, 'classical_time_s': classical_time,
    }

    return {
        'aer'        : {'counts': aer_counts, 'dm': aer_dm, 'codon_probs': aer_probs_counts, 'codon_probs_sv': aer_probs_sv, 'metrics': aer_metrics},
        'quebec'     : {'counts': quebec_counts, 'dm': quebec_dm, 'codon_probs': quebec_probs, 'metrics': quebec_metrics},
        'classical'  : {'codon_probs': pi_classical, 'P_t': P_t},
        'circuit_info': circuit_info,
        'trotter_info': trotter_info,
    }


# =========================================================================
# STEP 8: FIDELITY & VERIFICATION
# =========================================================================

def compute_evolution_fidelities(results, aae_step2_result=None):
    """
    Step 8: Compare quantum results to the classical exact answer.
    Uses Bhattacharyya coefficient as fidelity for probability distributions.
    """
    pi_classical  = results['classical']['codon_probs']
    pi_aer        = results['aer']['codon_probs']
    pi_quebec = results['quebec']['codon_probs']

    def dist_fidelity(p, q):
        return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)))) ** 2

    f_classical_aer        = dist_fidelity(pi_classical, pi_aer)
    f_classical_quebec     = dist_fidelity(pi_classical, pi_quebec)
    f_aer_quebec           = dist_fidelity(pi_aer, pi_quebec)

    tv_aer        = 0.5 * np.sum(np.abs(pi_classical - pi_aer))
    tv_quebec     = 0.5 * np.sum(np.abs(pi_classical - pi_quebec))

    aer_dm        = results['aer']['dm']
    quebec_dm     = results['quebec']['dm']

    try:
        f_dm_aer_quebec = state_fidelity(aer_dm, quebec_dm)
    except Exception:
        f_dm_aer_quebec = None

    if aae_step2_result is not None:
        initial_dm = aae_step2_result['initial_dm']
        try:
            f_initial_aer        = state_fidelity(initial_dm, aer_dm)
            f_initial_quebec     = state_fidelity(initial_dm, quebec_dm)
        except Exception:
            f_initial_aer = f_initial_quebec = None
    else:
        f_initial_aer = f_initial_quebec = None

    return {
        'f_classical_aer'        : f_classical_aer,
        'f_classical_quebec'     : f_classical_quebec,
        'f_aer_quebec'           : f_aer_quebec,
        'tv_classical_aer'       : tv_aer,
        'tv_classical_quebec'    : tv_quebec,
        'f_dm_aer_quebec'        : f_dm_aer_quebec,
        'f_initial_aer'          : f_initial_aer,
        'f_initial_quebec'       : f_initial_quebec,
        'trotter_error'          : 1.0 - f_classical_aer,
        'noise_error'            : f_classical_aer - f_classical_quebec,
    }


# =========================================================================
# PRINT REPORTS
# =========================================================================

def print_experiment_report(results, fidelities, sense_codons):
    """Print the full Step 7 + Step 8 experiment report."""
    ci = results['circuit_info']

    print("\n" + "=" * 70)
    print("  STEP 7: EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"\n  Setup:")
    print(f"    Evolution time t:       {ci['t']:.4f}")
    print(f"    Trotter steps (r):      {ci['n_trotter_steps']}")
    print(f"    Trotter order:          {ci['trotter_order']}")
    print(f"    Shots per backend:      {ci['shots']}")
    print(f"\n  Timing:")
    print(f"    Circuit build:          {ci['build_time_s']:.2f}s")
    print(f"    Transpile:              {ci['transpile_time_s']:.2f}s")
    print(f"    Aer simulation:         {ci['aer_run_time_s']:.2f}s")
    print(f"    Quebec simulation:      {ci['quebec_run_time_s']:.2f}s")
    print(f"    Classical reference:    {ci['classical_time_s']*1000:.2f}ms")
    print(f"\n  Transpiled circuit depth:")
    print(f"    Aer:        {results['aer']['metrics']['depth']}")
    print(f"    Quebec:     {results['quebec']['metrics']['depth']}")

    pi_cl  = results['classical']['codon_probs']
    pi_aer = results['aer']['codon_probs']
    pi_qb  = results['quebec']['codon_probs']

    print(f"\n  Codon distribution — top 10 by classical probability:")
    print(f"  {'Codon':>6}  {'Classical':>10}  {'Aer':>10}  {'Quebec':>11}  {'Δ(Aer)':>8}  {'Δ(Qbc)':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*8}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:10.6f}  {pi_aer[idx]:10.6f}  {pi_qb[idx]:11.6f}  {pi_aer[idx]-pi_cl[idx]:+8.5f}  {pi_qb[idx]-pi_cl[idx]:+8.5f}")

    f = fidelities
    print(f"\n" + "=" * 70)
    print(f"  STEP 8: VERIFICATION & FIDELITY")
    print(f"=" * 70)
    print(f"\n  Distribution fidelity (Bhattacharyya):")
    print(f"    F(classical, Aer)        = {f['f_classical_aer']:.6f}   <- Trotter error only")
    print(f"    F(classical, Quebec)     = {f['f_classical_quebec']:.6f}   <- Trotter + noise")
    print(f"    F(Aer, Quebec)           = {f['f_aer_quebec']:.6f}   <- pure noise impact")
    print(f"\n  Total variation distance (lower = better):")
    print(f"    TV(classical, Aer)        = {f['tv_classical_aer']:.6f}")
    print(f"    TV(classical, Quebec)     = {f['tv_classical_quebec']:.6f}")
    if f['f_dm_aer_quebec'] is not None:
        print(f"\n  Quantum state fidelity (density matrix):")
        print(f"    F_dm(Aer, Quebec)        = {f['f_dm_aer_quebec']:.6f}")
    if f['f_initial_aer'] is not None:
        print(f"\n  Evolution check (did the state actually change?):")
        print(f"    F(initial, Aer)          = {f['f_initial_aer']:.6f}   (< 1.0 means state evolved)")
        print(f"    F(initial, Quebec)       = {f['f_initial_quebec']:.6f}")
    print(f"\n  Error breakdown:")
    print(f"    Trotter approximation error: {f['trotter_error']:.6f}")
    print(f"    Hardware noise error:        {f['noise_error']:.6f}")
    print(f"    Total error:                 {1.0 - f['f_classical_quebec']:.6f}")

    fa, fs = f['f_classical_aer'], f['f_classical_quebec']
    print(f"\n  Verdict:")
    if fa > 0.95:   print(f"    Trotter accuracy:  EXCELLENT (F={fa:.4f})")
    elif fa > 0.90: print(f"    Trotter accuracy:  GOOD      (F={fa:.4f})")
    elif fa > 0.80: print(f"    Trotter accuracy:  FAIR      (F={fa:.4f}) — increase r")
    else:           print(f"    Trotter accuracy:  POOR      (F={fa:.4f}) — significantly increase r")
    if fs > 0.90:   print(f"    Noise resilience:  GOOD      (F={fs:.4f})")
    elif fs > 0.75: print(f"    Noise resilience:  MODERATE  (F={fs:.4f}) — circuit getting noisy")
    else:           print(f"    Noise resilience:  POOR      (F={fs:.4f}) — circuit too deep")


# =========================================================================
# Standalone execution has been removed.
# Use `python src/qsp_circuit.py` as the entry point for the GAPDH pipeline.
# =========================================================================

