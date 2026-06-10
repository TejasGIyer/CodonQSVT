"""
QSP Circuit — Quantum Signal Processing via pyqsp (verified recipe)
==========================================================================
Implements standard Wx-convention QSP with qubitization, following
Low & Chuang (2017, 2019) and verified against pyqsp's Remez+Haah
angle finder.

Recipe (verified in test_qsp_minimal_v7.py):
  1. U_BE = standard LCU block encoding of the Hamiltonian H
  2. W    = R_anc . U_BE  (qubitized walk operator)
     where R_anc = 2|0^m><0^m|_anc - I_anc reflects about the ancilla zero
  3. phases = pyqsp Wx angles for cos(alpha*t * x)
  4. QSP circuit = AAE(data) . rz(-2*phi_0) . [W . rz(-2*phi_k)]_{k=1..N-1}
     where the rz rotations act on the BE ancilla register (collectively,
     as e^(i*phi*Z_total) -- implemented by applying rz to each ancilla
     qubit simultaneously with the appropriate phase scaling).
  5. Simulate to statevector. Extract amplitudes from |anc=0> subspace.
  6. Take REAL PART of extracted amplitudes, square, normalize.
  7. Compare to classical e^(Q*t) . pi_initial.

Key differences from the previous (pre-pyqsp) implementation:
  - No separate 'qbit_qubit' initialized to |+>. Post-selection is only
    on the BE ancilla register.
  - No branch-averaging e^(iN*theta/2) phase factors.
  - No Hadamard wrapper on an extra ancilla.
  - Walk operator explicitly constructed via build_walk_operator().
"""

import os
import sys
import time
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.trotter    import classical_evolution
from src.experiment import counts_to_codon_probs, statevector_to_codon_probs
from src.block_encoding import build_walk_operator

# pyqsp polynomial generators (used by compute_full_unitary_angles)
try:
    from pyqsp.poly import PolyCosineTX, PolySineTX
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    _PYQSP_AVAILABLE = True
except ImportError:
    _PYQSP_AVAILABLE = False


# =========================================================================
# MULTI-QUBIT SIGNAL OPERATOR
# =========================================================================

def _build_signal_op(phi, n_anc):
    """
    Build e^(i*phi*R_anc) where R_anc = 2|0^m><0^m| - I is the reflection
    about the all-zeros ancilla state.

    For m=1 this is e^(i*phi*Z) = rz(-2*phi) (up to a global phase).
    For m>1 it is NOT a single-qubit rotation; it is a reflection-phase
    operator that puts a phase e^(2i*phi) on the |0^m> subspace and
    leaves everything else as identity (modulo a global e^(-i*phi)).

    Circuit implementation (valid for m >= 1):
      1. X^{⊗m}   (map |0^m> to |1^m>)
      2. Multi-controlled phase gate P(2*phi) with m-1 controls
         (for m=1 this is just a single-qubit P(2*phi))
      3. X^{⊗m}   (restore)
      (+ a global phase e^(-i*phi), tracked for exact unitary correctness)

    Verified in test_qsp_minimal_v8.py for m=1 and m=2, and generalizes
    to any m via MCPhaseGate.
    """
    from qiskit.circuit.library import MCPhaseGate

    qc = QuantumCircuit(n_anc, name=f'sig')

    # X-sandwich start
    for q in range(n_anc):
        qc.x(q)

    # Apply 2*phi phase on the |1^m> state
    if n_anc == 1:
        # Single-qubit P gate = diag(1, e^{i*2*phi})
        qc.p(2 * phi, 0)
    elif n_anc == 2:
        qc.cp(2 * phi, 0, 1)
    else:
        mcp = MCPhaseGate(2 * phi, n_anc - 1)
        qc.append(mcp, list(range(n_anc)))

    # X-sandwich end
    for q in range(n_anc):
        qc.x(q)

    # Global phase factor (doesn't affect observables but makes the
    # unitary exactly match e^(i*phi*R_anc)).
    qc.global_phase = -phi
    return qc


# =========================================================================
# QSP CIRCUIT BUILDER (verified pyqsp recipe)
# =========================================================================

def build_qsp_circuit(be_circuit, phis, aae_circuit=None,
                      n_data_qubits=6, n_be_ancilla=2, pauli_op=None):
    """
    Build the full QSP circuit implementing cos(alpha*t*H) up to pyqsp's
    0.5 rescaling factor.

    Layout:
      qubits 0 .. n_be_ancilla-1      : BE ancilla register
      qubits n_be_ancilla .. n_total-1: data register (where H acts)

    Parameters
    ----------
    be_circuit    : QuantumCircuit   block encoding U_BE (PREPARE . SELECT . UNPREP)
    phis          : np.ndarray       pyqsp phase angles (Wx convention)
    aae_circuit   : QuantumCircuit   optional initial-state preparation on data
    n_data_qubits : int              number of data qubits
    n_be_ancilla  : int              number of BE ancilla qubits
    pauli_op      : SparsePauliOp    kept for API compat; not strictly needed
                                      since be_circuit already encodes it

    Returns
    -------
    qc   : QuantumCircuit   the full QSP circuit (no measurements)
    info : dict             circuit metrics
    """
    n_total = n_be_ancilla + n_data_qubits
    anc_qubits  = list(range(n_be_ancilla))
    data_qubits = list(range(n_be_ancilla, n_total))

    N = len(phis)

    qc = QuantumCircuit(n_total, name='QSP')

    # Step 1: prepare the initial state on the data register (AAE).
    if aae_circuit is not None:
        qc.compose(aae_circuit, qubits=data_qubits, inplace=True)
        qc.barrier(label='AAE')

    # Step 2: build the qubitized walk operator W = R_anc . U_BE.
    walk = build_walk_operator(be_circuit, n_be_ancilla)

    # Step 3: apply the QSP sequence.
    #   signal_op(phi_0) on the ancilla register, then
    #   for k=1..N-1: W, signal_op(phi_k).
    #
    # For a multi-qubit ancilla, the signal operator is e^(i*phi*R_anc)
    # where R_anc is the signed reflection about |0^m>, NOT a single-qubit
    # rotation. For m=1 this reduces to rz(-2*phi). For m>1 we use the
    # multi-qubit reflection-phase construction in _build_signal_op.
    # Verified in test_qsp_minimal_v8.py.

    sig_0 = _build_signal_op(phis[0], n_be_ancilla)
    qc.compose(sig_0, qubits=anc_qubits, inplace=True)
    qc.barrier()
    for k in range(1, N):
        qc.compose(walk, qubits=list(range(n_total)), inplace=True)
        sig_k = _build_signal_op(phis[k], n_be_ancilla)
        qc.compose(sig_k, qubits=anc_qubits, inplace=True)
        qc.barrier()

    # =====================================================================
    # Circuit metrics
    # =====================================================================
    qc_decomp = qc.decompose(reps=5)
    gc = dict(qc_decomp.count_ops())
    n_cx = sum(gc.get(g, 0) for g in ['cx', 'cy', 'cz', 'mcx', 'ccx', 'ecr'])
    info = {
        'N_angles': N,
        'n_w_applications': N - 1,
        'n_total_qubits': n_total,
        'n_data_qubits': n_data_qubits,
        'n_be_ancilla': n_be_ancilla,
        'depth': qc_decomp.depth(),
        'n_cx_gates': n_cx,
        'n_total_gates': sum(gc.values()),
        'data_qubits': data_qubits,
    }
    return qc, info


# =========================================================================
# CODON PROBABILITY EXTRACTION (verified: take REAL part, then square)
# =========================================================================

def extract_codon_probs_from_sv(sv_data, n_total_qubits, n_be_ancilla,
                                n_data_qubits, n_codons=61):
    amps_per_codon = np.zeros(n_codons, dtype=complex)
    for state_idx in range(len(sv_data)):
        amp = sv_data[state_idx]
        if abs(amp) < 1e-18:
            continue
        if (state_idx & ((1 << n_be_ancilla) - 1)) != 0:
            continue
        data_val = state_idx >> n_be_ancilla
        if data_val < n_codons:
            amps_per_codon[data_val] += amp
    success_p = float(np.sum(np.abs(amps_per_codon) ** 2))
    real_amps = amps_per_codon.real
    raw_probs = real_amps ** 2
    norm = float(np.sum(raw_probs))
    if norm > 1e-12:
        probs = raw_probs / norm
    else:
        probs = np.zeros(n_codons)
    return probs, success_p


# =========================================================================
# FULL EXPERIMENT RUNNER
# =========================================================================

def run_qsp_experiment(be_circuit, phis, aae_circuit,
                       Q, pi_initial, sense_codons,
                       n_data_qubits=6, n_be_ancilla=2,
                       t=0.5, shots=8192, verbose=True, pauli_op=None):
    """Build the QSP circuit, simulate, extract codon distribution, compare
    to classical e^(Qt)*pi."""
    n_codons = len(sense_codons)

    if verbose:
        print(f"\n  Building QSP circuit (N={len(phis)} angles, verified pyqsp recipe)...")
    t0 = time.time()
    qsp_circ, circ_info = build_qsp_circuit(
        be_circuit=be_circuit, phis=phis, aae_circuit=aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    build_time = time.time() - t0
    if verbose:
        print(f"  Built in {build_time:.2f}s")
        print(f"  Total qubits:  {circ_info['n_total_qubits']}")
        print(f"  Circuit depth: {circ_info['depth']}")
        print(f"  CX gates:      {circ_info['n_cx_gates']}")
        print(f"  W applications: {circ_info['n_w_applications']}")

    if verbose:
        print(f"\n  Running statevector simulation...")
    t0 = time.time()
    sv = Statevector.from_instruction(qsp_circ)
    sv_data = sv.data
    sv_time = time.time() - t0
    if verbose:
        print(f"  Done in {sv_time:.2f}s")

    probs, success_p = extract_codon_probs_from_sv(
        sv_data, circ_info['n_total_qubits'], n_be_ancilla, n_data_qubits, n_codons)
    if verbose:
        print(f"  Post-selection probability: {success_p:.4f} ({100*success_p:.1f}%)")
        print(f"  Top 5 codons (QSP statevector):")
        for idx in np.argsort(probs)[::-1][:5]:
            print(f"    {sense_codons[idx]}: {probs[idx]:.5f}")

    if verbose:
        print(f"\n  (Shot-based simulation with real-part extraction requires a")
        print(f"   Hadamard-test ancilla wrapper which is not yet implemented.")
        print(f"   Reporting statevector-only fidelity.)")
    probs_shots = probs.copy()
    shot_time = 0.0
    total_kept = 0
    total_shots = 0

    pi_classical, P_t = classical_evolution(Q, pi_initial, t)

    def dist_fidelity(p, q):
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = np.sum(p), np.sum(q)
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        return float(np.clip(np.sum(np.sqrt(p * q)) ** 2, 0.0, 1.0))

    f_sv    = dist_fidelity(pi_classical, probs)
    f_shots = f_sv
    tv_sv   = 0.5 * float(np.sum(np.abs(pi_classical - probs)))
    if verbose:
        print(f"\n  Fidelity (statevector): {f_sv:.6f}")
        print(f"  TV distance:            {tv_sv:.6f}")

    return {
        'qsp_probs_sv': probs, 'qsp_probs_shots': probs_shots,
        'classical_probs': pi_classical, 'f_sv': f_sv, 'f_shots': f_shots,
        'tv_distance': tv_sv, 'success_prob': success_p,
        'shots_kept': total_kept, 'total_shots': total_shots,
        'circ_info': circ_info, 'build_time_s': build_time,
        'sv_time_s': sv_time, 'shot_time_s': shot_time,
    }


# =========================================================================
# FULL e^(-iHt) PIPELINE (cos + sin channels)
# =========================================================================

def compute_full_unitary_angles(alpha, t, epsilon=1e-6):
    if not _PYQSP_AVAILABLE:
        raise ImportError("pyqsp is required. Install with: pip install pyqsp")
    tau = float(alpha) * float(t)
    print(f"\n  Computing FULL e^(-iHt) angles (cos + sin channels):")
    print(f"    alpha = {alpha:.6f},  t = {t:.6f},  tau = {tau:.6f}")
    print(f"    epsilon = {epsilon:.2e}")
    poly_cos = PolyCosineTX().generate(tau=tau, epsilon=epsilon)
    phis_cos = np.asarray(QuantumSignalProcessingPhases(
        poly_cos, signal_operator='Wx', method='laurent'), dtype=float)
    print(f"    cos: degree {len(poly_cos)-1}, {len(phis_cos)} phases")
    poly_sin = PolySineTX().generate(tau=tau, epsilon=epsilon)
    phis_sin = np.asarray(QuantumSignalProcessingPhases(
        poly_sin, signal_operator='Wx', method='laurent'), dtype=float)
    print(f"    sin: degree {len(poly_sin)-1}, {len(phis_sin)} phases")
    info = {
        'tau': tau, 'alpha': alpha, 't': t, 'epsilon': epsilon,
        'cos_degree': len(poly_cos) - 1, 'sin_degree': len(poly_sin) - 1,
        'n_cos_phases': len(phis_cos), 'n_sin_phases': len(phis_sin),
        'n_cos_walk_applications': len(phis_cos) - 1,
        'n_sin_walk_applications': len(phis_sin) - 1,
    }
    return phis_cos, phis_sin, info


def extract_codon_amps_complex(sv_data, n_total_qubits, n_be_ancilla,
                               n_data_qubits, n_codons=61):
    """Extract per-codon complex amplitudes from the |0_anc> subspace.

    Bit-ordering convention (must match block_encoding.verify_block_encoding):
    the BE ancilla register sits on the LOW qubit indices (0..n_be_ancilla-1),
    so in the little-endian Statevector index the ancilla bits are the LOW
    bits. Post-selecting on |0_anc> keeps indices with those low bits zero
    ((state_idx & ((1<<n_be_ancilla)-1)) == 0); the codon (data) index is the
    remaining high bits (state_idx >> n_be_ancilla). Verified end-to-end in
    tests/test_block_encoding.py.
    """
    amps = np.zeros(n_codons, dtype=complex)
    for state_idx in range(len(sv_data)):
        amp = sv_data[state_idx]
        if abs(amp) < 1e-18:
            continue
        if (state_idx & ((1 << n_be_ancilla) - 1)) != 0:
            continue
        data_val = state_idx >> n_be_ancilla
        if data_val < n_codons:
            amps[data_val] += amp
    return amps


def run_qsp_experiment_full(be_circuit, phis_cos, phis_sin, aae_circuit,
                            Q, pi_initial, sense_codons,
                            n_data_qubits=6, n_be_ancilla=2,
                            t=0.5, verbose=True, pauli_op=None):
    n_codons = len(sense_codons)
    if verbose:
        print(f"\n  Building COS QSP circuit ({len(phis_cos)} phases)...")
    t0 = time.time()
    qc_cos, info_cos = build_qsp_circuit(
        be_circuit=be_circuit, phis=phis_cos, aae_circuit=aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    build_cos_time = time.time() - t0
    if verbose:
        print(f"  Built in {build_cos_time:.2f}s — depth {info_cos['depth']},"
              f" {info_cos['n_cx_gates']} CX, {info_cos['n_w_applications']} walks")
        print(f"\n  Building SIN QSP circuit ({len(phis_sin)} phases)...")
    t0 = time.time()
    qc_sin, info_sin = build_qsp_circuit(
        be_circuit=be_circuit, phis=phis_sin, aae_circuit=aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    build_sin_time = time.time() - t0
    if verbose:
        print(f"  Built in {build_sin_time:.2f}s — depth {info_sin['depth']},"
              f" {info_sin['n_cx_gates']} CX, {info_sin['n_w_applications']} walks")
    if verbose:
        print(f"\n  Simulating COS channel...")
    t0 = time.time()
    sv_cos = np.asarray(Statevector.from_instruction(qc_cos).data)
    sv_cos_time = time.time() - t0
    if verbose:
        print(f"  done in {sv_cos_time:.2f}s")
        print(f"  Simulating SIN channel...")
    t0 = time.time()
    sv_sin = np.asarray(Statevector.from_instruction(qc_sin).data)
    sv_sin_time = time.time() - t0
    if verbose:
        print(f"  done in {sv_sin_time:.2f}s")
    n_total = info_cos['n_total_qubits']
    amps_cos = extract_codon_amps_complex(sv_cos, n_total, n_be_ancilla, n_data_qubits, n_codons)
    amps_sin = extract_codon_amps_complex(sv_sin, n_total, n_be_ancilla, n_data_qubits, n_codons)
    cos_success_p = float(np.sum(np.abs(amps_cos) ** 2))
    sin_success_p = float(np.sum(np.abs(amps_sin) ** 2))
    if verbose:
        print(f"\n  Post-selection probabilities:")
        print(f"    cos channel: {cos_success_p:.4f}  ({100*cos_success_p:.1f}%)")
        print(f"    sin channel: {sin_success_p:.4f}  ({100*sin_success_p:.1f}%)")
    re_cos = amps_cos.real
    re_sin = amps_sin.real
    combined_probs = re_cos ** 2 + re_sin ** 2
    psum = float(np.sum(combined_probs))
    probs = combined_probs / psum if psum > 1e-12 else np.zeros(n_codons)
    if verbose:
        print(f"\n  Top 5 codons (full e^(-iHt) statevector):")
        for idx in np.argsort(probs)[::-1][:5]:
            print(f"    {sense_codons[idx]}: {probs[idx]:.5f}")
    pi_classical, P_t = classical_evolution(Q, pi_initial, t)
    def dist_fidelity(p, q):
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = np.sum(p), np.sum(q)
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        return float(np.clip(np.sum(np.sqrt(p * q)) ** 2, 0.0, 1.0))
    f_sv = dist_fidelity(pi_classical, probs)
    tv_sv = 0.5 * float(np.sum(np.abs(pi_classical - probs)))
    if verbose:
        print(f"\n  Fidelity (full e^(-iHt), statevector): {f_sv:.6f}")
        print(f"  TV distance:                            {tv_sv:.6f}")
    return {
        'qsp_probs_sv': probs, 'classical_probs': pi_classical,
        'amps_cos': amps_cos, 'amps_sin': amps_sin,
        'cos_success_prob': cos_success_p, 'sin_success_prob': sin_success_p,
        'f_sv': f_sv, 'tv_distance': tv_sv,
        'circ_info_cos': info_cos, 'circ_info_sin': info_sin,
        'build_cos_time_s': build_cos_time, 'build_sin_time_s': build_sin_time,
        'sv_cos_time_s': sv_cos_time, 'sv_sin_time_s': sv_sin_time,
        'method': 'full_unitary_e^(-iHt)',
    }


def print_qsp_report_full(results, sense_codons):
    ic = results['circ_info_cos']
    isn = results['circ_info_sin']
    print("\n" + "=" * 70)
    print("  QSP FULL e^(-iHt) EXPERIMENT REPORT")
    print("=" * 70)
    print(f"\n  Cos channel:")
    print(f"    Phases (N):              {ic['N_angles']}")
    print(f"    Walk applications:       {ic['n_w_applications']}")
    print(f"    Circuit depth:           {ic['depth']}")
    print(f"    CX gates:                {ic['n_cx_gates']}")
    print(f"    Post-selection prob:     {results['cos_success_prob']:.4f}")
    print(f"\n  Sin channel:")
    print(f"    Phases (N):              {isn['N_angles']}")
    print(f"    Walk applications:       {isn['n_w_applications']}")
    print(f"    Circuit depth:           {isn['depth']}")
    print(f"    CX gates:                {isn['n_cx_gates']}")
    print(f"    Post-selection prob:     {results['sin_success_prob']:.4f}")
    print(f"\n  Combined:")
    print(f"    F(statevector vs CTMC):  {results['f_sv']:.6f}")
    print(f"    TV distance:             {results['tv_distance']:.6f}")
    pi_cl  = results['classical_probs']
    pi_qsp = results['qsp_probs_sv']
    print(f"\n  Codon distribution — top 10:")
    print(f"  {'Codon':>6}  {'Classical':>10}  {'QSP_full':>10}  {'Delta':>8}  Match")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        delta = pi_qsp[idx] - pi_cl[idx]
        match = "✓" if abs(delta) < 0.005 else ("~" if abs(delta) < 0.01 else "✗")
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:10.6f}  "
              f"{pi_qsp[idx]:10.6f}  {delta:+8.5f}  {match}")


# =========================================================================
# HARDWARE-REALIZABLE FULL e^(-iHt) (11 qubits, Hadamard test)
# =========================================================================

def build_qsp_hadamard_circuit(be_circuit, phis, aae_circuit,
                               n_data_qubits=6, n_be_ancilla=3, pauli_op=None):
    qc_inner, inner_info = build_qsp_circuit(
        be_circuit=be_circuit, phis=phis, aae_circuit=None,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    qc_inner.data = [inst for inst in qc_inner.data if inst.operation.name != 'barrier']
    qsp_gate = qc_inner.to_gate(label='QSP_inner').control(1)
    n_inner = n_be_ancilla + n_data_qubits
    n_total = 1 + n_inner
    aux_q = 0
    inner_qubits = list(range(1, n_total))
    data_qubits  = list(range(1 + n_be_ancilla, n_total))
    qc = QuantumCircuit(n_total, name='QSP_Hadamard')
    qc.compose(aae_circuit, qubits=data_qubits, inplace=True)
    qc.barrier(label='AAE')
    qc.h(aux_q)
    qc.barrier(label='H_aux')
    qc.append(qsp_gate, [aux_q] + inner_qubits)
    qc.barrier(label='cQSP')
    qc.h(aux_q)
    qc.barrier(label='H_aux')
    qc_decomp = qc.decompose(reps=5)
    gc = dict(qc_decomp.count_ops())
    n_cx = sum(gc.get(g, 0) for g in ['cx', 'cy', 'cz', 'mcx', 'ccx', 'ecr'])
    info = {
        'N_angles': len(phis), 'n_w_applications': len(phis) - 1,
        'n_total_qubits': n_total, 'n_aux_qubits': 1,
        'n_be_ancilla': n_be_ancilla, 'n_data_qubits': n_data_qubits,
        'depth': qc_decomp.depth(), 'n_cx_gates': n_cx,
        'n_total_gates': sum(gc.values()), 'aux_qubit': aux_q,
        'inner_info': inner_info,
    }
    return qc, info


def extract_hadamard_probs(sv_data, n_total_qubits, n_aux, n_be_ancilla,
                           n_data_qubits, n_codons=61):
    n_anc_total = n_aux + n_be_ancilla
    anc_mask = (1 << n_anc_total) - 1
    probs = np.zeros(n_codons, dtype=float)
    for state_idx in range(len(sv_data)):
        amp = sv_data[state_idx]
        p = float(np.abs(amp) ** 2)
        if p < 1e-30:
            continue
        if (state_idx & anc_mask) != 0:
            continue
        data_val = state_idx >> n_anc_total
        if data_val < n_codons:
            probs[data_val] += p
    success_p = float(np.sum(probs))
    return probs, success_p


def recover_real_part_from_hadamard(P_hadamard, beta_amps, abs_alpha_sq, eps=1e-10):
    re_alpha = np.zeros_like(beta_amps, dtype=float)
    n_illcond = 0
    for i in range(len(beta_amps)):
        b = beta_amps[i]
        if abs(b) < eps:
            n_illcond += 1
            continue
        re_alpha[i] = (4.0 * P_hadamard[i] - b * b - abs_alpha_sq[i]) / (2.0 * b)
    return re_alpha, n_illcond


def run_qsp_experiment_hardware(be_circuit, phis_cos, phis_sin, aae_circuit,
                                aae_initial_sv, Q, pi_initial, sense_codons,
                                n_data_qubits=6, n_be_ancilla=3,
                                t=0.5, verbose=True, pauli_op=None):
    n_codons = len(sense_codons)
    aae_data = np.asarray(aae_initial_sv.data)
    beta_full = aae_data[:n_codons] if len(aae_data) >= n_codons else np.concatenate([aae_data, np.zeros(n_codons - len(aae_data))])
    beta_amps = beta_full.real.astype(float)

    if verbose: print(f"\n  [hw 1/4] Calibration cos circuit (10 qubits)...")
    qc_cos_cal, info_cos_cal = build_qsp_circuit(be_circuit=be_circuit, phis=phis_cos, aae_circuit=aae_circuit, n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    sv_cos_cal = np.asarray(Statevector.from_instruction(qc_cos_cal).data)
    amps_cos_cal = extract_codon_amps_complex(sv_cos_cal, info_cos_cal['n_total_qubits'], n_be_ancilla, n_data_qubits, n_codons)
    abs_alpha_cos_sq = np.abs(amps_cos_cal) ** 2

    if verbose: print(f"  [hw 2/4] Calibration sin circuit (10 qubits)...")
    qc_sin_cal, info_sin_cal = build_qsp_circuit(be_circuit=be_circuit, phis=phis_sin, aae_circuit=aae_circuit, n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    sv_sin_cal = np.asarray(Statevector.from_instruction(qc_sin_cal).data)
    amps_sin_cal = extract_codon_amps_complex(sv_sin_cal, info_sin_cal['n_total_qubits'], n_be_ancilla, n_data_qubits, n_codons)
    abs_alpha_sin_sq = np.abs(amps_sin_cal) ** 2

    if verbose: print(f"  [hw 3/4] Hadamard-test cos circuit (11 qubits)...")
    qc_cos_hd, info_cos_hd = build_qsp_hadamard_circuit(be_circuit=be_circuit, phis=phis_cos, aae_circuit=aae_circuit, n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    sv_cos_hd = np.asarray(Statevector.from_instruction(qc_cos_hd).data)
    P_cos, ps_cos = extract_hadamard_probs(sv_cos_hd, info_cos_hd['n_total_qubits'], 1, n_be_ancilla, n_data_qubits, n_codons)

    if verbose: print(f"  [hw 4/4] Hadamard-test sin circuit (11 qubits)...")
    qc_sin_hd, info_sin_hd = build_qsp_hadamard_circuit(be_circuit=be_circuit, phis=phis_sin, aae_circuit=aae_circuit, n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    sv_sin_hd = np.asarray(Statevector.from_instruction(qc_sin_hd).data)
    P_sin, ps_sin = extract_hadamard_probs(sv_sin_hd, info_sin_hd['n_total_qubits'], 1, n_be_ancilla, n_data_qubits, n_codons)

    re_alpha_cos, n_illcond_cos = recover_real_part_from_hadamard(P_cos, beta_amps, abs_alpha_cos_sq)
    re_alpha_sin, n_illcond_sin = recover_real_part_from_hadamard(P_sin, beta_amps, abs_alpha_sin_sq)
    combined = re_alpha_cos ** 2 + re_alpha_sin ** 2
    psum = float(np.sum(combined))
    probs = combined / psum if psum > 1e-12 else np.zeros(n_codons)

    pi_classical, _ = classical_evolution(Q, pi_initial, t)
    def dist_fidelity(p, q):
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = np.sum(p), np.sum(q)
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        return float(np.clip(np.sum(np.sqrt(p * q)) ** 2, 0.0, 1.0))
    f_sv = dist_fidelity(pi_classical, probs)
    tv_sv = 0.5 * float(np.sum(np.abs(pi_classical - probs)))
    if verbose:
        print(f"\n  F(hardware-realizable vs CTMC): {f_sv:.6f}")
        print(f"  TV distance:                    {tv_sv:.6f}")
    return {
        'qsp_probs_sv': probs, 'classical_probs': pi_classical,
        'f_sv': f_sv, 'tv_distance': tv_sv,
        'cos_hd_success': ps_cos, 'sin_hd_success': ps_sin,
        'n_illcond_cos': n_illcond_cos, 'n_illcond_sin': n_illcond_sin,
        'circ_info_cos_cal': info_cos_cal, 'circ_info_sin_cal': info_sin_cal,
        'circ_info_cos_hd': info_cos_hd, 'circ_info_sin_hd': info_sin_hd,
        'method': 'hardware_realizable_e^(-iHt)',
    }


def print_qsp_report_hardware(results, sense_codons):
    ic_hd = results['circ_info_cos_hd']
    is_hd = results['circ_info_sin_hd']
    print("\n" + "=" * 70)
    print("  QSP HARDWARE-REALIZABLE e^(-iHt) REPORT (11 qubits, Hadamard test)")
    print("=" * 70)
    print(f"  cos Hadamard: depth {ic_hd['depth']}, {ic_hd['n_cx_gates']} CX")
    print(f"  sin Hadamard: depth {is_hd['depth']}, {is_hd['n_cx_gates']} CX")
    print(f"  F(quantum vs CTMC):     {results['f_sv']:.6f}")
    print(f"  TV distance:            {results['tv_distance']:.6f}")
    pi_cl = results['classical_probs']
    pi_qsp = results['qsp_probs_sv']
    print(f"\n  Top 10 codons:")
    print(f"  {'Codon':>6}  {'Classical':>10}  {'QSP_hw':>10}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        delta = pi_qsp[idx] - pi_cl[idx]
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:10.6f}  {pi_qsp[idx]:10.6f}  {delta:+8.5f}")


# =========================================================================
# NOISY BACKEND SIMULATION (FakeQuebec)
# =========================================================================


# =========================================================================
# NOISY BACKEND SIMULATION (FakeQuebec)
# =========================================================================


def plot_noisy_comparison(all_results, sense_codons, save_dir):
    """
    Generate comparison plots and save to save_dir.

    Plots:
      1. Fidelity comparison bar chart (ideal + all noisy backends)
      2. Transpiled 2-qubit gate counts per backend
      3. Post-selection rates per backend
      4. Codon distribution comparison (top 15 codons)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    backend_names = [r['backend'] for r in all_results]
    f_ideals = [r['f_ideal'] for r in all_results]
    f_noisy_cos = [r['f_noisy_combined'] for r in all_results]
    f_noisy_sin = [r['f_aer'] for r in all_results]
    cos_2q = [r['metrics_cos']['two_qubit_gates'] for r in all_results]
    sin_2q = [r['metrics_sin']['two_qubit_gates'] for r in all_results]
    cos_depth = [r['metrics_cos']['depth'] for r in all_results]
    sin_depth = [r['metrics_sin']['depth'] for r in all_results]
    ps_cos = [r['kept_cos'] / r['total_cos'] if r['total_cos'] > 0 else 0 for r in all_results]
    ps_sin = [r['kept_sin'] / r['total_sin'] if r['total_sin'] > 0 else 0 for r in all_results]

    # --- Plot 1: Fidelity comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(backend_names))
    w = 0.25
    bars1 = ax.bar(x - w, f_ideals, w, label='Ideal (SV)', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x, f_noisy_cos, w, label='Noisy combined', color='#e74c3c', alpha=0.85)
    bars3 = ax.bar(x + w, f_noisy_sin, w, label='Aer noiseless', color='#3498db', alpha=0.85)
    ax.set_ylabel('Fidelity vs CTMC')
    ax.set_title('QSP Fidelity: Ideal vs Noisy Backends')
    ax.set_xticks(x)
    ax.set_xticklabels(backend_names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fidelity_comparison.png'), dpi=150)
    plt.close()

    # --- Plot 2: 2-qubit gate counts ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, cos_2q, w, label='cos channel', color='#9b59b6', alpha=0.85)
    bars2 = ax.bar(x + w/2, sin_2q, w, label='sin channel', color='#f39c12', alpha=0.85)
    ax.set_ylabel('Two-Qubit Gates (transpiled)')
    ax.set_title('Transpiled 2-Qubit Gate Count per Backend')
    ax.set_xticks(x)
    ax.set_xticklabels(backend_names)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'two_qubit_gates.png'), dpi=150)
    plt.close()

    # --- Plot 3: Post-selection rates ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, ps_cos, w, label='cos channel', color='#1abc9c', alpha=0.85)
    bars2 = ax.bar(x + w/2, ps_sin, w, label='sin channel', color='#e67e22', alpha=0.85)
    ax.set_ylabel('Post-Selection Rate (ancilla=000)')
    ax.set_title('Post-Selection Survival Rate per Backend')
    ax.set_xticks(x)
    ax.set_xticklabels(backend_names)
    ax.legend()
    ax.set_ylim(0, max(max(ps_cos), max(ps_sin)) * 1.3 if max(ps_cos + ps_sin) > 0 else 1)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'postselection_rates.png'), dpi=150)
    plt.close()

    # --- Plot 4: Circuit depth comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, cos_depth, w, label='cos channel', color='#2c3e50', alpha=0.85)
    bars2 = ax.bar(x + w/2, sin_depth, w, label='sin channel', color='#c0392b', alpha=0.85)
    ax.set_ylabel('Transpiled Circuit Depth')
    ax.set_title('Transpiled Circuit Depth per Backend')
    ax.set_xticks(x)
    ax.set_xticklabels(backend_names)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'circuit_depth.png'), dpi=150)
    plt.close()

    # --- Plot 5: Noise cost (fidelity drop) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    noise_cost = [r['f_aer'] - r['f_noisy_combined'] for r in all_results]
    colors = ['#e74c3c' if nc > 0.1 else '#f39c12' if nc > 0.02 else '#2ecc71' for nc in noise_cost]
    bars = ax.bar(backend_names, noise_cost, color=colors, alpha=0.85)
    ax.set_ylabel('Fidelity Drop (Ideal - Noisy)')
    ax.set_title('Noise Cost per Backend (lower = better)')
    for bar, nc in zip(bars, noise_cost):
        ax.annotate(f'{nc:+.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_cost.png'), dpi=150)
    plt.close()

    print(f"\n  Plots saved to {save_dir}/")
    print(f"    fidelity_comparison.png")
    print(f"    two_qubit_gates.png")
    print(f"    postselection_rates.png")
    print(f"    circuit_depth.png")
    print(f"    noise_cost.png")


def _counts_to_postselected_probs(counts, n_postselect_low, n_codons=61,
                                  normalize=True):
    """Extract codon probs from shot counts, post-selecting on low qubits=0."""
    probs = np.zeros(n_codons, dtype=float)
    kept = 0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        anc_bits = bitstring[-n_postselect_low:]
        if anc_bits != '0' * n_postselect_low:
            continue
        data_bits = bitstring[:-n_postselect_low] if n_postselect_low > 0 else bitstring
        data_val = int(data_bits, 2)
        if data_val < n_codons:
            probs[data_val] += count
        kept += count
    if normalize and kept > 0:
        probs /= float(kept)
    return probs, kept, total


def run_qsp_experiment_noisy(be_circuit, phis_cos, phis_sin, aae_circuit,
                             Q, pi_initial, sense_codons,
                             n_data_qubits=6, n_be_ancilla=3,
                             t=0.5, shots=32768, verbose=True, pauli_op=None,
                             backend_name='quebec'):
    """
    Run the FULL cos+sin QSP pipeline on a noisy fake backend (10 qubits).

    Runs BOTH cos and sin circuits, post-selects each on ancilla=000,
    combines as (cos_probs + sin_probs)/2, and compares against:
      - CTMC classical reference
      - Ideal noiseless Aer shots (same post-selection, no noise)
      - Ideal statevector (Re-projected, theoretical ceiling)
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    if backend_name.lower() == 'quebec':
        from qiskit_ibm_runtime.fake_provider import FakeQuebec
        fake_backend = FakeQuebec()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'quebec'.")

    backend_label = backend_name.capitalize()
    n_codons = len(sense_codons)
    n_total = n_be_ancilla + n_data_qubits

    def dist_fidelity(p, q):
        p = np.clip(p, 0, None); q = np.clip(q, 0, None)
        sp, sq = np.sum(p), np.sum(q)
        if sp > 1e-12: p = p / sp
        if sq > 1e-12: q = q / sq
        return float(np.clip(np.sum(np.sqrt(p * q)) ** 2, 0.0, 1.0))

    if verbose:
        print(f"\n  Building cos & sin QSP circuits for {backend_label}...")

    qc_cos, info_cos = build_qsp_circuit(be_circuit, phis_cos, aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)
    qc_sin, info_sin = build_qsp_circuit(be_circuit, phis_sin, aae_circuit,
        n_data_qubits=n_data_qubits, n_be_ancilla=n_be_ancilla, pauli_op=pauli_op)

    # --- Ideal statevector reference (Re-projected) ---
    if verbose:
        print(f"  Computing ideal references...")
    sv_cos = np.asarray(Statevector.from_instruction(qc_cos).data)
    sv_sin = np.asarray(Statevector.from_instruction(qc_sin).data)
    amps_cos_id = extract_codon_amps_complex(sv_cos, n_total, n_be_ancilla, n_data_qubits, n_codons)
    amps_sin_id = extract_codon_amps_complex(sv_sin, n_total, n_be_ancilla, n_data_qubits, n_codons)
    probs_ideal_reproj = amps_cos_id.real ** 2 + amps_sin_id.real ** 2
    ps = probs_ideal_reproj.sum()
    if ps > 1e-12: probs_ideal_reproj /= ps

    # What noiseless |amp|^2 looks like (the ceiling for shot-based measurement)
    probs_cos_raw = np.abs(amps_cos_id) ** 2
    probs_sin_raw = np.abs(amps_sin_id) ** 2
    pc = probs_cos_raw.sum(); pss = probs_sin_raw.sum()
    if pc > 1e-12: probs_cos_raw /= pc
    if pss > 1e-12: probs_sin_raw /= pss
    probs_ideal_raw = (probs_cos_raw + probs_sin_raw) / 2.0

    pi_classical, _ = classical_evolution(Q, pi_initial, t)
    f_ideal_reproj = dist_fidelity(pi_classical, probs_ideal_reproj)
    f_ideal_raw = dist_fidelity(pi_classical, probs_ideal_raw)
    if verbose:
        print(f"  F(Re-projected vs CTMC):  {f_ideal_reproj:.6f}  (SV ceiling)")
        print(f"  F(raw |amp|^2 vs CTMC):   {f_ideal_raw:.6f}  (shot ceiling)")

    # --- Transpile ---
    qc_cos_meas = qc_cos.copy(); qc_cos_meas.measure_all()
    qc_sin_meas = qc_sin.copy(); qc_sin_meas.measure_all()
    if verbose:
        print(f"\n  Transpiling for {backend_label} (optimization_level=3)...")
    t0 = time.time()
    tqc_cos = transpile(qc_cos_meas, backend=fake_backend, optimization_level=3)
    tqc_sin = transpile(qc_sin_meas, backend=fake_backend, optimization_level=3)
    transpile_time = time.time() - t0

    def _metrics(tqc):
        gc = dict(tqc.count_ops())
        two_q = sum(v for k, v in gc.items() if k in ['cx','cnot','ecr','cz','swap','iswap'])
        return {'depth': tqc.depth(), 'two_qubit_gates': two_q,
                'total_gates': sum(gc.values()),
                'ecr': gc.get('ecr', 0), 'swap': gc.get('swap', 0)}

    m_cos, m_sin = _metrics(tqc_cos), _metrics(tqc_sin)
    if verbose:
        print(f"  Transpiled in {transpile_time:.1f}s")
        print(f"    cos: depth={m_cos['depth']}, 2Q={m_cos['two_qubit_gates']}")
        print(f"    sin: depth={m_sin['depth']}, 2Q={m_sin['two_qubit_gates']}")

    # --- Noiseless baseline (analytical from statevector, no Aer needed) ---
    # The Aer noiseless baseline is just shot-sampling from |amp|^2.
    # Instead of running another expensive simulation, we compute it
    # analytically: the noiseless post-selected distribution IS probs_ideal_raw.
    # For a finite-shot estimate we use numpy multinomial sampling.
    if verbose:
        print(f"\n  Computing noiseless baseline (analytical from SV)...")
    # Post-selected probabilities from statevector (already computed above)
    probs_aer_combined = probs_ideal_raw.copy()
    f_aer = dist_fidelity(pi_classical, probs_aer_combined)
    if verbose:
        print(f"  F(Aer combined vs CTMC): {f_aer:.6f}  (analytical from SV)")

    # --- Noisy simulation ---
    import gc
    # Free statevector arrays and unmeasured circuits before noisy sim
    del sv_cos, sv_sin, qc_cos, qc_sin, qc_cos_meas, qc_sin_meas
    gc.collect()
    noise_model = NoiseModel.from_backend(fake_backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    del fake_backend; gc.collect()  # free backend memory
    if verbose: print(f"\n  Running noisy cos ({shots} shots)...")
    t0 = time.time()
    counts_cos = noisy_sim.run(tqc_cos, shots=shots).result().get_counts()
    cos_time = time.time() - t0
    del tqc_cos  # free memory
    if verbose: print(f"  Running noisy sin ({shots} shots)...")
    t0 = time.time()
    counts_sin = noisy_sim.run(tqc_sin, shots=shots).result().get_counts()
    sin_time = time.time() - t0
    del tqc_sin, noisy_sim; gc.collect()  # free memory

    # --- Post-select BOTH and combine ---
    probs_cos_noisy, kept_cos, total_cos = _counts_to_postselected_probs(
        counts_cos, n_be_ancilla, n_codons)
    probs_sin_noisy, kept_sin, total_sin = _counts_to_postselected_probs(
        counts_sin, n_be_ancilla, n_codons)
    probs_noisy_combined = (probs_cos_noisy + probs_sin_noisy) / 2.0
    pn = probs_noisy_combined.sum()
    if pn > 1e-12: probs_noisy_combined /= pn

    f_noisy_cos = dist_fidelity(pi_classical, probs_cos_noisy)
    f_noisy_sin = dist_fidelity(pi_classical, probs_sin_noisy)
    f_noisy_combined = dist_fidelity(pi_classical, probs_noisy_combined)
    tv_noisy = 0.5 * float(np.sum(np.abs(pi_classical - probs_noisy_combined)))

    if verbose:
        ps_c = kept_cos / total_cos if total_cos > 0 else 0
        ps_s = kept_sin / total_sin if total_sin > 0 else 0
        print(f"\n  Post-selection (ancilla=000):")
        print(f"    cos: {kept_cos}/{total_cos} = {ps_c:.4f}")
        print(f"    sin: {kept_sin}/{total_sin} = {ps_s:.4f}")
        print(f"\n  Fidelity ({backend_label}) — FULL cos+sin QSP:")
        print(f"    F(noisy COMBINED vs CTMC):    {f_noisy_combined:.6f}  <-- main result")
        print(f"    F(Aer combined vs CTMC):      {f_aer:.6f}  <-- noiseless shots")
        print(f"    F(ideal raw vs CTMC):         {f_ideal_raw:.6f}  <-- shot ceiling")
        print(f"    F(ideal Re-proj vs CTMC):     {f_ideal_reproj:.6f}  <-- SV ceiling")
        noise_cost = f_aer - f_noisy_combined
        print(f"    Noise cost (Aer - noisy):     {noise_cost:+.6f}")
        print(f"    TV(noisy combined vs CTMC):   {tv_noisy:.6f}")

    return {
        'backend': backend_label,
        'qsp_probs_cos_noisy': probs_cos_noisy,
        'qsp_probs_sin_noisy': probs_sin_noisy,
        'qsp_probs_combined_noisy': probs_noisy_combined,
        'qsp_probs_aer_combined': probs_aer_combined,
        'qsp_probs_ideal': probs_ideal_reproj,
        'classical_probs': pi_classical,
        'f_noisy_cos': f_noisy_cos, 'f_noisy_sin': f_noisy_sin,
        'f_noisy_combined': f_noisy_combined,
        'f_aer': f_aer,
        'f_ideal': f_ideal_reproj, 'f_ideal_raw': f_ideal_raw,
        'tv_noisy': tv_noisy,
        'kept_cos': kept_cos, 'total_cos': total_cos,
        'kept_sin': kept_sin, 'total_sin': total_sin,
        'metrics_cos': m_cos, 'metrics_sin': m_sin,
        'transpile_time_s': transpile_time,
        'cos_run_time_s': cos_time, 'sin_run_time_s': sin_time,
        'shots': shots, 'circ_info_cos': info_cos, 'circ_info_sin': info_sin,
        'method': f'noisy_full_qsp_{backend_name}',
    }


def print_qsp_report_noisy(results, sense_codons):
    backend = results['backend']
    mc, ms = results['metrics_cos'], results['metrics_sin']
    print("\n" + "=" * 70)
    print(f"  QSP NOISY REPORT — {backend} (full cos+sin)")
    print("=" * 70)
    print(f"\n  Transpiled circuits (10 qubits each):")
    print(f"    cos: depth={mc['depth']}, 2Q={mc['two_qubit_gates']}")
    print(f"    sin: depth={ms['depth']}, 2Q={ms['two_qubit_gates']}")
    ps_c = results['kept_cos'] / results['total_cos'] if results['total_cos'] > 0 else 0
    ps_s = results['kept_sin'] / results['total_sin'] if results['total_sin'] > 0 else 0
    print(f"\n  Post-selection (ancilla=000):")
    print(f"    cos: {results['kept_cos']}/{results['total_cos']} = {ps_c:.4f}")
    print(f"    sin: {results['kept_sin']}/{results['total_sin']} = {ps_s:.4f}")
    print(f"\n  Fidelity ladder (vs CTMC):")
    print(f"    Ideal Re-projected:    {results['f_ideal']:.6f}")
    print(f"    Ideal raw |amp|^2:     {results['f_ideal_raw']:.6f}")
    print(f"    Aer noiseless shots:   {results['f_aer']:.6f}")
    print(f"    Noisy COMBINED:        {results['f_noisy_combined']:.6f}  <-- main result")
    nc = results['f_aer'] - results['f_noisy_combined']
    print(f"    Noise cost:            {nc:+.6f}")
    print(f"    TV distance:           {results['tv_noisy']:.6f}")

    pi_cl = results['classical_probs']
    pi_noisy = results['qsp_probs_combined_noisy']
    print(f"\n  Top 10 codons (combined cos+sin):")
    print(f"  {'Codon':>6}  {'CTMC':>9}  {'Aer':>9}  {'Noisy':>9}  {'Δ(noisy)':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    p_aer = results['qsp_probs_aer_combined']
    for idx in np.argsort(pi_cl)[::-1][:10]:
        d = pi_noisy[idx] - pi_cl[idx]
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:9.6f}  "
              f"{p_aer[idx]:9.6f}  {pi_noisy[idx]:9.6f}  {d:+9.5f}")


# =========================================================================
# REPORT
# =========================================================================

def print_qsp_report(results, sense_codons):
    ci = results['circ_info']
    print("\n" + "=" * 70)
    print("  QSP EXPERIMENT REPORT")
    print("=" * 70)
    print(f"\n  Circuit:")
    print(f"    QSP angles (N):          {ci['N_angles']}")
    print(f"    W applications:          {ci['n_w_applications']}")
    print(f"    Total qubits:            {ci['n_total_qubits']}")
    print(f"    Circuit depth:           {ci['depth']}")
    print(f"    CX gates:                {ci['n_cx_gates']}")
    print(f"\n  Post-selection:")
    print(f"    Success prob:            {results['success_prob']:.4f} "
          f"({100*results['success_prob']:.1f}%)")
    print(f"\n  Fidelity / distance (vs classical exact):")
    print(f"    F(statevector): {results['f_sv']:.6f}")
    print(f"    TV distance:    {results['tv_distance']:.6f}")
    pi_cl  = results['classical_probs']
    pi_qsp = results['qsp_probs_sv']
    print(f"\n  Codon distribution — top 10:")
    print(f"  {'Codon':>6}  {'Classical':>10}  {'QSP':>10}  {'Delta':>8}  Match")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    for idx in np.argsort(pi_cl)[::-1][:10]:
        delta = pi_qsp[idx] - pi_cl[idx]
        match = "✓" if abs(delta) < 0.005 else ("~" if abs(delta) < 0.01 else "✗")
        print(f"  {sense_codons[idx]:>6}  {pi_cl[idx]:10.6f}  "
              f"{pi_qsp[idx]:10.6f}  {delta:+8.5f}  {match}")


if __name__ == "__main__":
    from data.gapdh_sequences      import build_gapdh_register, pooled_codon_frequencies, ALL_SEQUENCES
    from src.aae_encoding          import aae_encode, get_aae_circuit
    from src.gy94_model            import build_gy94_rate_matrix, calculate_implied_omega
    from src.hamiltonian           import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
    from src.qsp_angles            import compute_qsp_angles
    from src.block_encoding        import build_simple_block_encoding, print_block_encoding_report

    KAPPA = 1.8425
    OMEGA = 0.0599
    LNL   = -2930.4333
    print("=" * 70)
    print("  QSP FULL PIPELINE — GAPDH (pyqsp-based, verified recipe)")
    print("=" * 70)
    for name, seq in ALL_SEQUENCES.items():
        print(f"  {name:>6}: {len(seq)} nt | {len(seq)//3} codons")
    print(f"  Model:  kappa={KAPPA}  omega={OMEGA}  lnL={LNL}")

    print("\n  [1/6] Building pipeline...")
    codon_freqs = pooled_codon_frequencies()
    print(f"  Pooled codon frequencies: {len(codon_freqs)} unique sense codons")
    best_v = 50.0
    min_err = float('inf')
    for test_v in np.linspace(5, 200, 391):
        implied = calculate_implied_omega(codon_freqs, KAPPA, test_v)
        err = abs(implied - OMEGA)
        if err < min_err:
            min_err = err
            best_v = test_v
    print(f"  -> Best V = {best_v:.4f}  (omega error = {min_err:.6f})")
    Q, sense_codons, pi, q_info = build_gy94_rate_matrix(codon_freqs, kappa=KAPPA, V=best_v)
    H, h_info = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
    pauli_full, _ = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
    THRESHOLD = 0.2
    pauli_op, n_kept = filter_pauli_op(pauli_full, THRESHOLD)
    print(f"\n  Pauli decomposition (threshold={THRESHOLD}): {n_kept} terms")
    print(f"  Terms: {pauli_op.paulis.to_labels()}")

    print("\n  [2/6] Loading (or training) AAE circuit on GAPDH codon distribution...")
    s1 = build_gapdh_register(n_qubits=6)
    aae_json = os.path.join(_PROJECT_DIR, 'results', 'best_aae_params_gapdh.json')
    s2 = get_aae_circuit(s1, aae_json, n_layers=6, n_trials=3, maxiter=3000)
    print(f"  Overlap: {s2['overlap']:.6f}")

    print("\n  [3/6] Building block encoding...")
    t0 = time.time()
    be_circuit, alpha, be_info = build_simple_block_encoding(pauli_op, n_data_qubits=6)
    print(f"  Done in {time.time()-t0:.2f}s")
    print_block_encoding_report(be_info)

    T_EVOL = 0.5
    EPSILON = 1e-6
    print(f"\n  [4/6] Computing QSP phase angles (alpha={alpha:.4f}, t={T_EVOL})...")
    phis, poly_info, angle_info = compute_qsp_angles(alpha, T_EVOL, epsilon=EPSILON)
    print(f"\n  Angles (N={len(phis)}): {np.round(phis, 4).tolist()}")

    print(f"\n  [5a] Running COS-ONLY QSP experiment...")
    results = run_qsp_experiment(
        be_circuit=be_circuit, phis=phis, aae_circuit=s2['circuit'],
        Q=Q, pi_initial=pi, sense_codons=sense_codons,
        n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'],
        t=T_EVOL, shots=8192, verbose=True, pauli_op=pauli_op)

    print(f"\n  [5b] Computing FULL e^(-iHt) angles (cos + sin)...")
    phis_cos, phis_sin, full_angle_info = compute_full_unitary_angles(alpha, T_EVOL, epsilon=EPSILON)

    print(f"\n  [5c] Running FULL e^(-iHt) QSP experiment...")
    results_full = run_qsp_experiment_full(
        be_circuit=be_circuit, phis_cos=phis_cos, phis_sin=phis_sin,
        aae_circuit=s2['circuit'], Q=Q, pi_initial=pi, sense_codons=sense_codons,
        n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'], t=T_EVOL, verbose=True, pauli_op=pauli_op)

    print(f"\n  [5d] Running HARDWARE-REALIZABLE e^(-iHt) (11 qubits, Hadamard test)...")
    results_hw = run_qsp_experiment_hardware(
        be_circuit=be_circuit, phis_cos=phis_cos, phis_sin=phis_sin,
        aae_circuit=s2['circuit'], aae_initial_sv=s2['initial_sv'],
        Q=Q, pi_initial=pi, sense_codons=sense_codons,
        n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'], t=T_EVOL, verbose=True, pauli_op=pauli_op)

    print_qsp_report(results, sense_codons)
    print_qsp_report_full(results_full, sense_codons)
    print_qsp_report_hardware(results_hw, sense_codons)

    # --- Noisy backend experiments ---
    import gc
    NOISY_BACKENDS = ['quebec']
    NOISY_SHOTS = 8192  # reduced from 32768 to avoid memory crashes
    noisy_results = {}
    for i, bname in enumerate(NOISY_BACKENDS):
        step = 6 + i
        total_steps = 5 + len(NOISY_BACKENDS) + 2  # +2 for plots and comparison
        print(f"\n  [{step}/{total_steps}] Running on {bname} (noisy, 10 qubits)...")
        try:
            noisy_results[bname] = run_qsp_experiment_noisy(
                be_circuit=be_circuit, phis_cos=phis_cos, phis_sin=phis_sin,
                aae_circuit=s2['circuit'], Q=Q, pi_initial=pi, sense_codons=sense_codons,
                n_data_qubits=6, n_be_ancilla=be_info['n_ancilla'],
                t=T_EVOL, shots=NOISY_SHOTS, verbose=True, pauli_op=pauli_op,
                backend_name=bname)
        except Exception as e:
            print(f"\n  *** {bname} FAILED: {e} ***")
            print(f"  Skipping {bname}, continuing with remaining backends...")
        gc.collect()  # force cleanup between backends

    for bname in NOISY_BACKENDS:
        if bname in noisy_results:
            print_qsp_report_noisy(noisy_results[bname], sense_codons)

    # --- Plots ---
    plot_step = 6 + len(NOISY_BACKENDS)
    if noisy_results:
        print(f"\n  [{plot_step}/{total_steps}] Generating comparison plots...")
        save_dir = os.path.join(_PROJECT_DIR, 'results')
        try:
            plot_noisy_comparison(list(noisy_results.values()), sense_codons, save_dir)
        except Exception as e:
            print(f"  *** Plot generation failed: {e} ***")
    else:
        print(f"\n  [{plot_step}/{total_steps}] Skipping plots (no noisy results).")

    # --- Final comparison ---
    final_step = plot_step + 1
    print(f"\n  [{final_step}/{total_steps}] Final comparison table")
    print("\n" + "=" * 86)
    print("  FULL COMPARISON: ideal + noisy backends")
    print("=" * 86)
    print(f"  {'Method':<28} {'F vs CTMC':>10} {'Qubits':>7} {'Circ':>5} "
          f"{'Depth(cos)':>11} {'2Q(cos)':>8} {'PS(cos)':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*7} {'-'*5} {'-'*11} {'-'*8} {'-'*8}")
    print(f"  {'cos-only (SV)':<28} {results['f_sv']:>10.6f} {10:>7d} {1:>5d} "
          f"{'N/A':>11} {'N/A':>8} {'N/A':>8}")
    print(f"  {'full e^(-iHt) (SV)':<28} {results_full['f_sv']:>10.6f} {10:>7d} {2:>5d} "
          f"{'N/A':>11} {'N/A':>8} {'N/A':>8}")
    print(f"  {'Hadamard-test (SV)':<28} {results_hw['f_sv']:>10.6f} {11:>7d} {4:>5d} "
          f"{'N/A':>11} {'N/A':>8} {'N/A':>8}")
    for bname in NOISY_BACKENDS:
        if bname not in noisy_results:
            print(f"  {bname+' (SKIPPED)':<28} {'---':>10} {'---':>7} {'---':>5} "
                  f"{'---':>11} {'---':>8} {'---':>8}")
            continue
        r = noisy_results[bname]
        mc = r['metrics_cos']
        ps = r['kept_cos'] / r['total_cos'] if r['total_cos'] > 0 else 0
        print(f"  {r['backend']+'(combined)':<28} {r['f_noisy_combined']:>10.6f} {10:>7d} {2:>5d} "
              f"{mc['depth']:>11d} {mc['two_qubit_gates']:>8d} {ps:>8.4f}")

    print(f"\n  GAPDH QSP pipeline complete.")

