"""
Step 6: Trotterized Time Evolution Circuit
==========================================
Builds the quantum circuit that evolves an initial codon distribution
forward in time under the GY94 mutation Hamiltonian H.

The core idea:
    We want to apply  e^{-iHt}  to our quantum state.
    Since H = Σ_k c_k P_k  (sum of Pauli strings that don't commute),
    we use the first-order Trotter approximation:

        e^{-iHt} ≈ [ Π_k  e^{-i c_k P_k Δt} ]^r

    where  Δt = t / r  and  r = number of Trotter steps.

    Each factor  e^{-i c_k P_k Δt}  is a Pauli rotation gate that
    Qiskit can compile directly into CNOT + Rz sequences.

Outputs:
    - A QuantumCircuit implementing the Trotter evolution
    - Diagnostic info (gate count, depth, estimated Trotter error)

Reference: Nielsen & Chuang, "Quantum Computation and Quantum Information"
           Section 4.7 (Hamiltonian simulation via Trotterization)
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter


# =========================================================================
# CORE TROTTER CIRCUIT BUILDER
# =========================================================================

def build_trotter_circuit(pauli_op, t, n_trotter_steps=3, order=1):
    """
    Build the Trotterized time evolution circuit for e^{-iHt}.

    Parameters
    ----------
    pauli_op       : SparsePauliOp
        The Hamiltonian H expressed as a sum of Pauli strings (from Step 5).
    t              : float
        Total evolution time (in units of 1/mutation_rate).
    n_trotter_steps: int
        Number of Trotter steps r. More steps -> higher accuracy, deeper circuit.
        Recommended: start with 1-3 for noisy hardware, 5-10 for ideal simulation.
    order          : int
        Trotter order. 1 = first-order (LieTrotter), 2 = second-order (Suzuki).
        Order 2 is more accurate for the same depth but ~2x more gates.

    Returns
    -------
    trotter_circuit    : QuantumCircuit (with PauliEvolutionGate, not decomposed)
    trotter_decomposed : QuantumCircuit (fully decomposed into basic gates)
    info               : dict with diagnostic information
    """
    n_qubits = pauli_op.num_qubits

    # Choose Trotter synthesis method
    if order == 1:
        synthesis = LieTrotter(reps=n_trotter_steps)
    elif order == 2:
        synthesis = SuzukiTrotter(order=2, reps=n_trotter_steps)
    else:
        raise ValueError(f"Unsupported Trotter order: {order}. Use 1 or 2.")

    # Build the PauliEvolutionGate
    evolution_gate = PauliEvolutionGate(pauli_op, time=t, synthesis=synthesis)

    # Wrap in a QuantumCircuit
    trotter_circuit = QuantumCircuit(n_qubits)
    trotter_circuit.append(evolution_gate, range(n_qubits))

    # Decompose into basic gates for inspection
    trotter_decomposed = trotter_circuit.decompose(reps=3)

    gate_counts = dict(trotter_decomposed.count_ops())
    depth       = trotter_decomposed.depth()
    n_cx        = gate_counts.get('cx', 0) + gate_counts.get('ecr', 0)
    n_rz        = gate_counts.get('rz', 0)
    n_total     = sum(gate_counts.values())

    # Trotter error bound estimate
    # ||e^{-iHt} - Trotter|| <= (||H|| * t)^2 / (2r)  [first order]
    pauli_norm = float(np.sum(np.abs(pauli_op.coeffs)))
    if order == 1:
        trotter_error_bound = (pauli_norm * t) ** 2 / (2 * n_trotter_steps)
    else:
        trotter_error_bound = (pauli_norm * t) ** 3 / (12 * n_trotter_steps ** 2)

    info = {
        'n_qubits'           : n_qubits,
        't'                  : t,
        'n_trotter_steps'    : n_trotter_steps,
        'order'              : order,
        'n_pauli_terms'      : len(pauli_op.coeffs),
        'depth'              : depth,
        'n_cx_gates'         : n_cx,
        'n_rz_gates'         : n_rz,
        'n_total_gates'      : n_total,
        'gate_counts'        : gate_counts,
        'pauli_norm'         : pauli_norm,
        'trotter_error_bound': trotter_error_bound,
    }

    return trotter_circuit, trotter_decomposed, info


# =========================================================================
# FULL CIRCUIT: AAE PREP + TROTTER EVOLUTION
# =========================================================================

def build_full_evolution_circuit(aae_circuit, pauli_op, t, n_trotter_steps=3, order=1):
    """
    Combine AAE state preparation with Trotter time evolution.

    Circuit structure:
        |0...0> -> [AAE: prepare |psi_0>] -> [Trotter: e^{-iHt}] -> |psi(t)>

    Parameters
    ----------
    aae_circuit    : QuantumCircuit
    pauli_op       : SparsePauliOp
    t              : float
    n_trotter_steps: int
    order          : int

    Returns
    -------
    full_circuit       : QuantumCircuit  (no measurements, for statevector)
    full_circuit_meas  : QuantumCircuit  (with measurements, for shot-based sim)
    trotter_info       : dict
    """
    n_qubits = aae_circuit.num_qubits
    assert pauli_op.num_qubits == n_qubits, (
        f"Qubit mismatch: AAE has {n_qubits} qubits, "
        f"Hamiltonian has {pauli_op.num_qubits} qubits."
    )

    trotter_circuit, _, trotter_info = build_trotter_circuit(
        pauli_op, t, n_trotter_steps=n_trotter_steps, order=order
    )

    # No measurements (for statevector / fidelity)
    full_circuit = QuantumCircuit(n_qubits)
    full_circuit.compose(aae_circuit, inplace=True)
    full_circuit.barrier(label=f"t={t:.3f}")
    full_circuit.compose(trotter_circuit, inplace=True)

    # With measurements (for shot-based simulators)
    full_circuit_meas = full_circuit.copy()
    full_circuit_meas.measure_all()

    full_decomposed = full_circuit.decompose(reps=3)
    combined_gate_counts = dict(full_decomposed.count_ops())

    trotter_info['full_circuit_depth']       = full_decomposed.depth()
    trotter_info['full_circuit_total_gates'] = sum(combined_gate_counts.values())
    trotter_info['full_circuit_cx_gates']    = (
        combined_gate_counts.get('cx', 0) + combined_gate_counts.get('ecr', 0)
    )

    return full_circuit, full_circuit_meas, trotter_info


# =========================================================================
# TROTTER PARAMETER SWEEP
# =========================================================================

def sweep_trotter_steps(pauli_op, t, step_range=(1, 2, 3, 5), order=1):
    """
    Sweep over Trotter step counts and report circuit metrics.
    Use this to pick the right accuracy/depth tradeoff before hardware runs.
    """
    results = []
    print(f"\n  Trotter sweep: t={t:.3f}, order={order}")
    print(f"  {'Steps':>6}  {'Depth':>7}  {'CX gates':>9}  {'Error bound':>13}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*13}")

    for r in step_range:
        _, _, info = build_trotter_circuit(pauli_op, t, n_trotter_steps=r, order=order)
        print(f"  {r:>6}  {info['depth']:>7}  {info['n_cx_gates']:>9}  {info['trotter_error_bound']:>13.6f}")
        results.append({'n_steps': r, **info})

    return results


# =========================================================================
# CLASSICAL REFERENCE EVOLUTION (for Step 8 verification)
# =========================================================================

def classical_evolution(Q, pi_initial, t):
    """
    Classically compute the evolved codon distribution after time t.
    Uses scipy matrix exponential:  pi(t) = e^{Qt} @ pi(0)

    Parameters
    ----------
    Q          : np.ndarray (61, 61)
    pi_initial : np.ndarray (61,)
    t          : float

    Returns
    -------
    pi_t : np.ndarray (61,)
    P_t  : np.ndarray (61, 61)
    """
    import scipy.linalg
    P_t  = scipy.linalg.expm(Q * t)
    pi_t = P_t @ pi_initial
    pi_t = np.clip(pi_t, 0, None)
    pi_t /= pi_t.sum()
    return pi_t, P_t


# =========================================================================
# PRINT REPORT
# =========================================================================

def print_trotter_report(trotter_info, has_full_circuit=False):
    """Print a formatted summary of the Trotter circuit."""
    print("\n" + "=" * 70)
    print("  STEP 6: TROTTERIZED TIME EVOLUTION CIRCUIT")
    print("=" * 70)

    print(f"\n  Evolution parameters:")
    print(f"    Evolution time t:          {trotter_info['t']:.4f}")
    print(f"    Trotter steps (r):         {trotter_info['n_trotter_steps']}")
    print(f"    Trotter order:             {trotter_info['order']}")
    print(f"    Pauli terms in H:          {trotter_info['n_pauli_terms']}")
    print(f"    Pauli norm (||H||):        {trotter_info['pauli_norm']:.6f}")

    print(f"\n  Trotter circuit (evolution only):")
    print(f"    Circuit depth:             {trotter_info['depth']}")
    print(f"    CX / two-qubit gates:      {trotter_info['n_cx_gates']}")
    print(f"    Rz gates:                  {trotter_info['n_rz_gates']}")
    print(f"    Total gates:               {trotter_info['n_total_gates']}")
    print(f"    Trotter error bound:       {trotter_info['trotter_error_bound']:.6f}")

    if has_full_circuit:
        print(f"\n  Full circuit (AAE + Trotter):")
        print(f"    Circuit depth:             {trotter_info['full_circuit_depth']}")
        print(f"    CX / two-qubit gates:      {trotter_info['full_circuit_cx_gates']}")
        print(f"    Total gates:               {trotter_info['full_circuit_total_gates']}")

    eb = trotter_info['trotter_error_bound']
    if eb < 0.01:
        quality = "Excellent — suitable for hardware"
    elif eb < 0.05:
        quality = "Good — minor Trotter noise expected"
    elif eb < 0.15:
        quality = "Moderate — increase r for better accuracy"
    else:
        quality = "High error — increase r significantly"

    print(f"\n  Trotter quality: {quality}")
    print(f"    (Smaller error bound -> more accurate time evolution)")


# =========================================================================
# Standalone execution has been removed.
# Use `python src/qsp_circuit.py` as the entry point for the GAPDH pipeline.
# =========================================================================

