"""
test_qsp_minimal_v8.py — Multi-qubit BE ancilla test.

Earlier tests (v3, v5, v7) all used a 1-qubit BE ancilla, where the
signal operator e^(i*phi*Z) is just rz(-2*phi) on the single ancilla.
For GAPDH, the BE ancilla is 3 qubits (8 Pauli terms -> ceil(log2(8))=3),
and my first port to the project applied rz to just ONE of the three
ancilla qubits, which is structurally wrong.

The correct signal operator for an m-qubit ancilla in Wx-convention QSP
is:  e^(i*phi*R_anc)  where R_anc = 2|0^m><0^m| - I  (signed reflection).
This equals e^(i*phi) on |0^m> and e^(-i*phi) on every other state.

For m=1, R_anc = Z and we recover e^(i*phi*Z) = rz(-2*phi).
For m>1, R_anc is NOT a tensor product of single-qubit Pauli Zs; it's
a projector-based reflection.

This test verifies both:
  (a) we can build e^(i*phi*R_anc) correctly for m=2, and
  (b) the full QSP circuit with this signal operator produces cos(t*H)
      on a non-trivial H.

Test Hamiltonian: H = 0.3*XI + 0.3*IX + 0.2*ZI + 0.2*IZ
  4-term LCU, needs m=2 ancilla qubits, acts on 2 data qubits.
"""

import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.library import StatePreparation

from pyqsp import poly as pyqsp_poly
from pyqsp import angle_sequence as pyqsp_angseq

print("=" * 72)
print("  MINIMAL QSP TEST v8 — multi-qubit BE ancilla (m=2)")
print("=" * 72)

# =====================================================================
# Setup
# =====================================================================
TAU_PHYS = 1.0
EPSILON = 1e-6

# 2-data-qubit Hamiltonian with 4 Pauli terms
# H = 0.3*(XI + IX) + 0.2*(ZI + IZ)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

XI = np.kron(X, I2)
IX = np.kron(I2, X)
ZI = np.kron(Z, I2)
IZ = np.kron(I2, Z)

H_mat = 0.3*XI + 0.3*IX + 0.2*ZI + 0.2*IZ
coeffs = np.array([0.3, 0.3, 0.2, 0.2])  # 4 terms
ALPHA = float(np.sum(np.abs(coeffs)))    # 1.0
TAU_QSP = TAU_PHYS * ALPHA               # 1.0

print(f"  H = 0.3*XI + 0.3*IX + 0.2*ZI + 0.2*IZ")
print(f"  alpha (1-norm) = {ALPHA}")
print(f"  TAU_QSP = {TAU_QSP}")

# Target: cos(TAU_PHYS * H). Use scipy.linalg.cosm for the matrix cosine.
target_cos_tH = scipy.linalg.cosm(TAU_PHYS * H_mat)
expected_block = 0.5 * target_cos_tH  # pyqsp's 0.5 rescale

np.set_printoptions(precision=5, suppress=True, linewidth=140)
print(f"\n  Target cos(TAU_PHYS * H):")
print(target_cos_tH)

# =====================================================================
# Build the LCU block encoding of H with m=2 ancilla qubits
#
# Layout: q0, q1 = 2 ancilla qubits, q2, q3 = 2 data qubits (total 4)
#
# PREPARE: 4-amplitude state = [sqrt(0.3), sqrt(0.3), sqrt(0.2), sqrt(0.2)]/sqrt(alpha)
# SELECT: for k=0..3, controlled-P_k on data when ancilla = k (binary)
#   k=0 (00) -> XI: X on q2, I on q3
#   k=1 (01) -> IX: I on q2, X on q3
#   k=2 (10) -> ZI: Z on q2, I on q3
#   k=3 (11) -> IZ: I on q2, Z on q3
# UNPREPARE: inverse of PREPARE
# =====================================================================
prep_amps = np.sqrt(np.abs(coeffs) / ALPHA)
prep_amps /= np.linalg.norm(prep_amps)
prep_gate = StatePreparation(prep_amps)

n_anc = 2
n_data = 2
n_tot = n_anc + n_data

ube = QuantumCircuit(n_tot, name='U_BE')
ube.append(prep_gate, [0, 1])

# SELECT for each k
pauli_labels = ['XI', 'IX', 'ZI', 'IZ']
for k, label in enumerate(pauli_labels):
    k_bits = format(k, f'0{n_anc}b')
    # Flip ancillas to make |k> -> |11>
    for j in range(n_anc):
        if k_bits[n_anc-1-j] == '0':
            ube.x(j)
    # Controlled-Pauli from the now-|11> ancilla
    ctrls = [0, 1]
    for q_idx, p_char in enumerate(label):
        # label is 2 chars, q_idx=0 -> q2 (first data), q_idx=1 -> q3
        target = n_anc + q_idx
        if p_char == 'I':
            continue
        elif p_char == 'X':
            ube.mcx(ctrls, target)
        elif p_char == 'Z':
            ube.h(target)
            ube.mcx(ctrls, target)
            ube.h(target)
    # Flip ancillas back
    for j in range(n_anc):
        if k_bits[n_anc-1-j] == '0':
            ube.x(j)

ube.append(prep_gate.inverse(), [0, 1])

# Verify block encoding property
U_BE_mat = Operator(ube).data
# Extract the (anc=00, anc=00) block. In Qiskit ordering, basis state
# index decomposes as (q3 q2 q1 q0), with q0 being LSB. anc=00 means
# q0=0 AND q1=0, so the bottom 2 bits of the index are 0. That's indices
# 0, 4, 8, 12 (i.e., indices where index % 4 == 0).
indices_anc00 = [i for i in range(2**n_tot) if (i & ((1 << n_anc) - 1)) == 0]
inner_block = U_BE_mat[np.ix_(indices_anc00, indices_anc00)]
inner_err = float(np.max(np.abs(inner_block - H_mat / ALPHA)))
print(f"\n  U_BE block-encoding check: max |<00|U_BE|00> - H/alpha| = {inner_err:.4e}")
if inner_err > 1e-12:
    print(f"  ✗ Block encoding is broken.")
    print(f"  inner_block =\n{inner_block}")
    print(f"  H/alpha     =\n{H_mat / ALPHA}")
    import sys
    sys.exit(1)
print(f"  ✓ Block encoding verified.")

# =====================================================================
# Build the qubitized walk W = R_anc . U_BE with m=2
#
# R_anc = 2|00><00| - I on the ancilla register.
# For m=2, this is X^⊗2 . CZ . X^⊗2  (up to a global phase of -1 maybe,
# which doesn't matter for observables).
# =====================================================================
walk = QuantumCircuit(n_tot, name='walk')
walk.compose(ube, qubits=list(range(n_tot)), inplace=True)
# Reflection about |00>_anc
walk.x(0)
walk.x(1)
walk.cz(0, 1)
walk.x(0)
walk.x(1)

# Sanity: does this walk have the expected walk eigenstructure?
walk_mat = Operator(walk).data
walk_inner = walk_mat[np.ix_(indices_anc00, indices_anc00)]
print(f"\n  Walk's (anc=00) block vs H/alpha:")
print(f"  max err: {float(np.max(np.abs(walk_inner - H_mat/ALPHA))):.4e}")
# This check doesn't fully test the walk property, but confirms the
# reflection doesn't corrupt the inner block.

# =====================================================================
# Get pyqsp angles for cos(TAU_QSP * x)
# =====================================================================
poly_coeffs = pyqsp_poly.PolyCosineTX().generate(tau=TAU_QSP, epsilon=EPSILON)
phiset = np.asarray(pyqsp_angseq.QuantumSignalProcessingPhases(
    poly_coeffs, signal_operator='Wx', eps=EPSILON), dtype=float)
N_phi = len(phiset)
print(f"\n  pyqsp returned {N_phi} phases")

# =====================================================================
# Build the CORRECT multi-qubit signal operator: e^(i*phi*R_anc)
#
# R_anc = 2|00><00| - I on the m=2 ancilla register.
# e^(i*phi*R_anc) = e^(i*phi)*|00><00| + e^(-i*phi)*(I - |00><00|)
#                = e^(-i*phi) * (I + (e^(2i*phi) - 1)*|00><00|)
# Up to the global e^(-i*phi), this is the identity plus a phase
# kickback when the ancilla is in |00>.
#
# Circuit implementation (modulo global phase):
#   1. X on every ancilla qubit          (|00> -> |11>)
#   2. MCPhase(2*phi) on the ancillas    (apply e^(2i*phi) if all |1>)
#   3. X on every ancilla qubit          (|11> -> |00>)
#
# Then the overall circuit applies e^(2i*phi) on the |00> state and
# identity on every other state. To match e^(i*phi*R_anc) exactly we
# need to also apply e^(-i*phi) to all states; since global phases
# don't affect observables, we can skip this OR include it via a
# global_phase attribute.
# =====================================================================

def build_signal_op(phi, n_anc):
    """e^(i*phi*R_anc) where R_anc = 2|0^m><0^m| - I. Ignores global phase."""
    qc = QuantumCircuit(n_anc, name=f'sig({phi:.3f})')
    if n_anc == 1:
        # m=1: R_anc = Z, so e^(i*phi*Z) = rz(-2*phi) up to global phase
        qc.rz(-2 * phi, 0)
        return qc
    # m > 1: X-sandwich around an MC phase gate
    for q in range(n_anc):
        qc.x(q)
    # MCPhase(2*phi) on all n_anc qubits
    if n_anc == 2:
        # 2-qubit MCPhase: just CP(2*phi)
        qc.cp(2 * phi, 0, 1)
    else:
        # m>=3: general multi-controlled phase via mcp
        from qiskit.circuit.library import MCPhaseGate
        mcp = MCPhaseGate(2 * phi, n_anc - 1)
        qc.append(mcp, list(range(n_anc)))
    for q in range(n_anc):
        qc.x(q)
    # Global phase factor e^(-i*phi) -- observable-irrelevant but include
    # it for exact unitary correctness in the test
    qc.global_phase = -phi
    return qc


# Verify signal_op for m=2: it should equal e^(i*phi*R_anc) exactly.
print(f"\n  Verifying multi-qubit signal operator for m=2:")
for test_phi in [0.3, 0.7, -0.5]:
    so = build_signal_op(test_phi, 2)
    so_mat = Operator(so).data
    # Expected: e^(i*phi*R_anc) with R_anc = 2*|00><00| - I_4
    P00 = np.zeros((4, 4), dtype=complex); P00[0, 0] = 1.0
    R = 2 * P00 - np.eye(4, dtype=complex)
    expected_so = scipy.linalg.expm(1j * test_phi * R)
    err = float(np.max(np.abs(so_mat - expected_so)))
    print(f"    phi={test_phi:+.2f}: max err = {err:.4e}")
    assert err < 1e-12, f"signal_op for m=2, phi={test_phi} is wrong"
print(f"  ✓ Signal operator verified for m=2.")

# =====================================================================
# Build the full QSP circuit using the correct multi-qubit signal op
# =====================================================================
def build_qsp_multi(phiset, walk_circ, n_anc_):
    qc = QuantumCircuit(n_tot, name='qsp')
    # First signal op
    so0 = build_signal_op(phiset[0], n_anc_)
    qc.compose(so0, qubits=list(range(n_anc_)), inplace=True)
    # Then alternating walk + signal op
    for k in range(1, len(phiset)):
        qc.compose(walk_circ, qubits=list(range(n_tot)), inplace=True)
        sok = build_signal_op(phiset[k], n_anc_)
        qc.compose(sok, qubits=list(range(n_anc_)), inplace=True)
    return qc

qsp = build_qsp_multi(phiset, walk, n_anc)
U_qsp = Operator(qsp).data

# Extract the (anc=00, anc=00) block (4x4 on the data register)
qsp_block = U_qsp[np.ix_(indices_anc00, indices_anc00)]
print(f"\n  QSP (anc=00) block (should be complex, real part = 0.5*cos(t*H)):")
print(qsp_block)
print(f"\n  Expected 0.5*cos(t*H):")
print(expected_block)

real_err = float(np.max(np.abs(qsp_block.real - expected_block.real)))
imag_mag = float(np.max(np.abs(qsp_block.imag)))
print(f"\n  max |Re(block) - Re(expected)| = {real_err:.4e}")
print(f"  max |Im(block)|                = {imag_mag:.4e}  (auxiliary polynomial, expected nonzero)")

if real_err < 1e-3:
    print(f"\n  ★★★ SUCCESS — multi-qubit (m=2) QSP recipe verified.")
    print(f"\n  Next step: port build_signal_op() into the project's")
    print(f"  build_qsp_circuit() to replace the buggy single-rz approach.")
else:
    print(f"\n  ✗ Multi-qubit test failed. Real part of block doesn't match target.")

# =====================================================================
# Full workflow test: apply to non-trivial |psi_0> and compare probs
# =====================================================================
print(f"\n  ---- Full workflow test on |psi_0> = normalized(0.5, 0.3, 0.4, 0.6) ----")
psi0 = np.array([0.5, 0.3, 0.4, 0.6], dtype=complex)
psi0 /= np.linalg.norm(psi0)

psi_target = target_cos_tH @ psi0
prob_target = np.abs(psi_target) ** 2
prob_target /= np.sum(prob_target)

# Build circuit with AAE-like state prep
prep_psi = StatePreparation(psi0)
qc_full = QuantumCircuit(n_tot)
qc_full.append(prep_psi, [n_anc, n_anc+1])  # data qubits
qc_full.compose(qsp, inplace=True)
sv = np.asarray(Statevector.from_instruction(qc_full).data)

# Extract amps from anc=00 subspace, take real part, square, normalize
amps_ps = np.array([sv[i] for i in indices_anc00])
real_amps = amps_ps.real
prob_quantum = real_amps ** 2
prob_quantum /= np.sum(prob_quantum)

tv = 0.5 * np.sum(np.abs(prob_target - prob_quantum))
print(f"  prob_target  = {prob_target}")
print(f"  prob_quantum = {prob_quantum}")
print(f"  TV distance  = {tv:.4e}")
if tv < 1e-3:
    print(f"  ★ Full workflow passes for m=2.")
else:
    print(f"  ✗ Full workflow fails.")

print("\n" + "=" * 72)
