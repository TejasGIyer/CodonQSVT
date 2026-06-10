"""
Block Encoding — Embedding H inside a Unitary
==============================================
QSP requires the Hamiltonian H to be embedded inside a larger unitary
matrix U_H such that:

    <0| U_H |0> = H / alpha

where alpha = ||H|| (spectral norm) and |0> refers to the ancilla qubit(s).

LCU method: U_H = PREPARE† · SELECT · PREPARE

Reference:
    Childs & Wiebe (2012), Berry et al. (2019)
"""

import os
import sys
import time
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit.circuit.library import StatePreparation


# =========================================================================
# LCU BLOCK ENCODING
# =========================================================================

def build_lcu_block_encoding(pauli_op, n_data_qubits=6):
    """
    Build LCU block encoding: PREPARE† · SELECT · PREPARE

    Register layout (consistent throughout):
        qubits 0..n_ancilla-1:               ancilla register
        qubits n_ancilla..n_ancilla+n_data-1: data register

    FIX: Previous version had PREPARE on qubits 0..n_anc but SELECT
    controls on qubits n_data..n_data+n_anc (a mismatch). Now both
    use ancilla at the front (0..n_anc-1) consistently.
    """
    coeffs  = pauli_op.coeffs
    labels  = pauli_op.paulis.to_labels()
    n_terms = len(coeffs)

    abs_coeffs = np.abs(coeffs)
    alpha      = float(np.sum(abs_coeffs))

    n_ancilla = max(1, int(np.ceil(np.log2(n_terms))))
    n_ancilla_states = 2 ** n_ancilla
    n_total = n_ancilla + n_data_qubits

    print(f"  Block encoding setup:")
    print(f"    Pauli terms:           {n_terms}")
    print(f"    Alpha (1-norm):        {alpha:.4f}")
    print(f"    Ancilla qubits needed: {n_ancilla}  ({n_ancilla_states} states)")
    print(f"    Data qubits:           {n_data_qubits}")
    print(f"    Total qubits:          {n_total}")

    prepare_amplitudes = np.zeros(n_ancilla_states)
    prepare_amplitudes[:n_terms] = np.sqrt(abs_coeffs / alpha)
    norm = np.linalg.norm(prepare_amplitudes)
    if norm > 0:
        prepare_amplitudes /= norm

    prepare_gate = StatePreparation(prepare_amplitudes)

    # Build SELECT circuit with CONSISTENT register layout:
    # ancilla = qubits 0..n_ancilla-1, data = qubits n_ancilla..n_total-1
    select_circ = QuantumCircuit(n_total, name='SELECT')

    for k, (label, coeff) in enumerate(zip(labels, coeffs)):
        if k >= n_ancilla_states:
            break

        k_binary = format(k, f'0{n_ancilla}b')

        # FIX: Ancilla qubits are at indices 0..n_ancilla-1 (not n_data..n_data+n_anc)
        flip_qubits = [j for j in range(n_ancilla) if k_binary[n_ancilla-1-j] == '0']
        for q in flip_qubits:
            select_circ.x(q)

        # Controls = all ancilla qubits (0..n_ancilla-1)
        ctrl_qubits = list(range(n_ancilla))
        for qubit_idx, pauli_char in enumerate(reversed(label)):
            # FIX: Data qubits start at index n_ancilla
            data_qubit = n_ancilla + qubit_idx
            if data_qubit >= n_total:
                break
            if pauli_char == 'I':
                continue
            elif pauli_char == 'X':
                select_circ.mcx(ctrl_qubits, data_qubit)
            elif pauli_char == 'Y':
                select_circ.sdg(data_qubit)
                select_circ.mcx(ctrl_qubits, data_qubit)
                select_circ.s(data_qubit)
            elif pauli_char == 'Z':
                select_circ.h(data_qubit)
                select_circ.mcx(ctrl_qubits, data_qubit)
                select_circ.h(data_qubit)

        for q in flip_qubits:
            select_circ.x(q)

    # Assemble: PREPARE on ancilla, then SELECT on all, then PREPARE† on ancilla
    be_circuit = QuantumCircuit(n_total, name='BlockEncoding')
    be_circuit.append(prepare_gate, range(n_ancilla))
    be_circuit.compose(select_circ, range(n_total), inplace=True)
    be_circuit.append(prepare_gate.inverse(), range(n_ancilla))

    be_decomposed = be_circuit.decompose(reps=3)
    gate_counts   = dict(be_decomposed.count_ops())
    depth         = be_decomposed.depth()
    n_cx          = sum(gate_counts.get(g, 0) for g in ['cx','cy','cz','ccx','mcx'])

    info = {
        'n_terms': n_terms, 'n_ancilla': n_ancilla,
        'n_data_qubits': n_data_qubits, 'n_total_qubits': n_total,
        'alpha': alpha, 'depth': depth, 'n_cx_gates': n_cx,
        'n_total_gates': sum(gate_counts.values()), 'gate_counts': gate_counts,
        'success_prob': 1.0 / (alpha ** 2),
    }
    return be_circuit, alpha, info


# =========================================================================
# VERIFY BLOCK ENCODING
# =========================================================================

def verify_block_encoding(be_circuit, H_matrix, alpha, n_data_qubits=6, n_ancilla=None):
    """Verify <0_anc| U_BE |0_anc> = H / alpha.

    Bit-ordering convention (CRITICAL)
    ----------------------------------
    Both block-encoding builders place the ancilla register FIRST, i.e. on
    the LOW qubit indices (0..n_ancilla-1) and the data register on the HIGH
    indices (n_ancilla..n_total-1). Qiskit's Statevector/Operator index is
    little-endian in qubit index, so the ancilla bits are the LOW bits of the
    integer state index. The |0_anc> subspace is therefore the set of state
    indices whose low n_ancilla bits are zero:  (idx & (2^n_anc - 1)) == 0,
    and the data value is  idx >> n_ancilla.  This MUST match the extraction
    masks in qsp_circuit.extract_codon_amps_complex /
    extract_codon_probs_from_sv. The unit test in
    tests/test_block_encoding.py pins this end to end.

    n_ancilla defaults to the value implied by the circuit width when not
    given (n_total - n_data_qubits), removing the old hard-coded 2.
    """
    if n_ancilla is None:
        n_ancilla = be_circuit.num_qubits - n_data_qubits
    n_total = n_data_qubits + n_ancilla
    n_states_data = 2 ** n_data_qubits
    n_states_total = 2 ** n_total
    anc_mask = (1 << n_ancilla) - 1

    if n_total > 12:
        print(f"  Verification skipped: too large ({n_total} qubits)")
        return None
    try:
        U = Operator(be_circuit).data
    except Exception as e:
        print(f"  Verification failed: {e}")
        return None

    # |0_anc> rows/cols: low n_ancilla bits of the index are zero (little-endian)
    anc_zero_idx = [i for i in range(n_states_total) if (i & anc_mask) == 0]
    H_block = np.zeros((n_states_data, n_states_data), dtype=complex)
    for i, ri in enumerate(anc_zero_idx):
        for j, cj in enumerate(anc_zero_idx):
            H_block[i, j] = U[ri, cj]

    # complex dtype: H may have imaginary entries (Pauli Y terms). A real
    # buffer would silently discard them and corrupt the comparison.
    H_padded = np.zeros((n_states_data, n_states_data), dtype=complex)
    H_padded[:H_matrix.shape[0], :H_matrix.shape[1]] = H_matrix
    return float(np.max(np.abs(H_block - H_padded / alpha)))


# =========================================================================
# SIMPLIFIED BLOCK ENCODING (for small threshold, few terms)
# =========================================================================

def build_simple_block_encoding(pauli_op, n_data_qubits=6):
    """
    Simplified block encoding for <=16 Pauli terms.
    Uses QuantumRegister for consistent naming. Already correct.
    """
    coeffs  = np.array(pauli_op.coeffs, dtype=complex)
    labels  = pauli_op.paulis.to_labels()
    n_terms = len(coeffs)

    if n_terms > 16:
        print(f"  Warning: {n_terms} terms is large for simple block encoding.")

    abs_coeffs = np.abs(coeffs)
    alpha      = float(np.sum(abs_coeffs))
    n_ancilla  = max(1, int(np.ceil(np.log2(max(n_terms, 2)))))

    n_anc_states = 2 ** n_ancilla
    amps = np.zeros(n_anc_states)
    amps[:n_terms] = np.sqrt(abs_coeffs / alpha)
    amps /= np.linalg.norm(amps)

    anc  = QuantumRegister(n_ancilla, 'anc')
    data = QuantumRegister(n_data_qubits, 'data')
    be_circuit = QuantumCircuit(anc, data, name='BE_simple')

    prep_gate = StatePreparation(amps)
    be_circuit.append(prep_gate, anc)

    for k in range(min(n_terms, n_anc_states)):
        label  = labels[k]
        k_bits = format(k, f'0{n_ancilla}b')
        ctrl_q = list(range(n_ancilla))

        for j in range(n_ancilla):
            if k_bits[n_ancilla-1-j] == '0':
                be_circuit.x(anc[j])

        for q_idx, p_char in enumerate(reversed(label)):
            if q_idx >= n_data_qubits: break
            if p_char == 'X':
                be_circuit.mcx(ctrl_q, n_ancilla + q_idx)
            elif p_char == 'Y':
                be_circuit.sdg(n_ancilla + q_idx)
                be_circuit.mcx(ctrl_q, n_ancilla + q_idx)
                be_circuit.s(n_ancilla + q_idx)
            elif p_char == 'Z':
                be_circuit.h(n_ancilla + q_idx)
                be_circuit.mcx(ctrl_q, n_ancilla + q_idx)
                be_circuit.h(n_ancilla + q_idx)

        for j in range(n_ancilla):
            if k_bits[n_ancilla-1-j] == '0':
                be_circuit.x(anc[j])

    be_circuit.append(prep_gate.inverse(), anc)

    be_decomposed = be_circuit.decompose(reps=3)
    gate_counts   = dict(be_decomposed.count_ops())

    info = {
        'n_terms': n_terms, 'n_ancilla': n_ancilla,
        'n_data_qubits': n_data_qubits,
        'n_total_qubits': n_data_qubits + n_ancilla,
        'alpha': alpha, 'depth': be_decomposed.depth(),
        'n_total_gates': sum(gate_counts.values()),
        'gate_counts': gate_counts,
        'success_prob': 1.0 / (alpha ** 2),
    }
    return be_circuit, alpha, info


# =========================================================================
# QUBITIZED WALK OPERATOR
# =========================================================================

def build_walk_operator(be_circuit, n_ancilla):
    """
    Build the qubitized walk operator W = R_anc . U_BE.

    R_anc = 2|0^m><0^m|_anc - I_anc  is the reflection about the
    all-zeros state of the ancilla register. For m=1 this is Pauli Z.
    For m>1 this is a multi-controlled Z preceded/followed by X gates
    on all ancillas.

    The walk operator W has the key property that its eigenvalues are
    e^(±i*arccos(lambda)) for each eigenvalue lambda of the inner operator
    H/alpha = <0^m|U_BE|0^m>. This is the structure QSP expects as its
    signal operator in the Wx convention.

    Parameters
    ----------
    be_circuit : QuantumCircuit   the LCU block encoding U_BE
    n_ancilla  : int              number of BE ancilla qubits (m)

    Returns
    -------
    walk : QuantumCircuit  same qubit layout as be_circuit, with the
                           reflection appended after U_BE
    """
    from qiskit import QuantumCircuit
    walk = QuantumCircuit(*be_circuit.qregs, name='walk')
    walk.compose(be_circuit, inplace=True)

    # Get the ancilla qubits from the first register (block_encoding.py
    # places the ancilla register first when building with QuantumRegister).
    anc_qubits = list(walk.qubits[:n_ancilla])

    if n_ancilla == 1:
        # Reflection about |0>: 2|0><0| - I = Z
        walk.z(anc_qubits[0])
    else:
        # Reflection about |0^m>: X_all . MCZ . X_all
        # (X-flip all ancillas, multi-controlled-Z, X-flip back)
        for q in anc_qubits:
            walk.x(q)
        # MCZ on all m ancilla qubits: use h-mcx-h on the last qubit
        # with the others as controls.
        if len(anc_qubits) == 2:
            walk.cz(anc_qubits[0], anc_qubits[1])
        else:
            last = anc_qubits[-1]
            ctrls = anc_qubits[:-1]
            walk.h(last)
            walk.mcx(ctrls, last)
            walk.h(last)
        for q in anc_qubits:
            walk.x(q)

    return walk


def print_block_encoding_report(info, verified_error=None):
    print("\n" + "=" * 70)
    print("  BLOCK ENCODING REPORT")
    print("=" * 70)
    print(f"\n  Pauli terms encoded:   {info['n_terms']}")
    print(f"  Ancilla qubits:        {info['n_ancilla']}")
    print(f"  Data qubits:           {info['n_data_qubits']}")
    print(f"  Total qubits:          {info['n_total_qubits']}")
    print(f"  Alpha (1-norm):        {info['alpha']:.6f}")
    print(f"  Circuit depth:         {info['depth']}")
    print(f"  Total gates:           {info['n_total_gates']}")
    print(f"  Post-selection prob:   {info['success_prob']:.4f} ({100*info['success_prob']:.2f}%)")
    print(f"  Expected shots needed: {int(1/info['success_prob'])} per successful measurement")
    if verified_error is not None:
        status = "PASS" if verified_error < 1e-6 else "WARN"
        print(f"\n  Verification error:    {verified_error:.2e}  [{status}]")


# =========================================================================
# Standalone execution has been removed.
# Use `python src/qsp_circuit.py` as the entry point for the GAPDH pipeline.
# =========================================================================

