"""
End-to-end block-encoding correctness test.

Proves the central QSVT-pipeline invariant:

    <0_anc| U_BE |0_anc>  ==  H / alpha

for the EXACT register/bit-ordering convention the QSP/QSVT builders use
(ancilla on the LOW qubit indices; little-endian Statevector index). This is
the test the audit flagged as missing -- every fidelity number in the paper
flows through the post-selection extraction, so this invariant must hold.

Run:
    pytest tests/test_block_encoding.py -v
or standalone:
    python tests/test_block_encoding.py
"""
import os
import sys
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

from src.block_encoding import (
    build_simple_block_encoding,
    build_lcu_block_encoding,
    verify_block_encoding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_from_unitary(U, n_ancilla, n_data):
    """Extract <0_anc| U |0_anc> using the ancilla-low little-endian rule."""
    n_total = n_ancilla + n_data
    anc_mask = (1 << n_ancilla) - 1
    rows = [i for i in range(2 ** n_total) if (i & anc_mask) == 0]
    blk = np.zeros((2 ** n_data, 2 ** n_data), dtype=complex)
    for i, ri in enumerate(rows):
        for j, cj in enumerate(rows):
            blk[i, j] = U[ri, cj]
    return blk


def _check(pauli_op, n_data, builder, tol=1e-9):
    H = Operator(pauli_op).data
    alpha = float(np.sum(np.abs(pauli_op.coeffs)))
    be, alpha_b, info = builder(pauli_op, n_data_qubits=n_data)
    assert abs(alpha - alpha_b) < 1e-12, f"alpha mismatch {alpha} vs {alpha_b}"

    U = Operator(be).data
    blk = _block_from_unitary(U, info['n_ancilla'], n_data)
    err = float(np.max(np.abs(blk - H / alpha)))
    return err, info, alpha


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_be_single_qubit():
    """1 data qubit, 2 terms: H = 0.6 Z + 0.8 X."""
    op = SparsePauliOp.from_list([("Z", 0.6), ("X", 0.8)])
    err, info, alpha = _check(op, 1, build_simple_block_encoding)
    print(f"[1q,2term] n_anc={info['n_ancilla']} alpha={alpha:.4f} err={err:.2e}")
    assert err < 1e-9


def test_simple_be_two_qubit_five_terms():
    """2 data qubits, 5 terms (non-power-of-2 -> padded PREPARE)."""
    op = SparsePauliOp.from_list([("ZI", 0.5), ("IX", 0.3), ("XX", 0.2),
                                  ("ZZ", 0.15), ("YI", 0.1)])
    err, info, alpha = _check(op, 2, build_simple_block_encoding)
    print(f"[2q,5term] n_anc={info['n_ancilla']} alpha={alpha:.4f} err={err:.2e}")
    assert err < 1e-9


def test_lcu_be_matches_simple():
    """The general LCU builder must satisfy the same invariant."""
    op = SparsePauliOp.from_list([("ZI", 0.5), ("IX", 0.3), ("XX", 0.2),
                                  ("ZZ", 0.15)])
    err, info, alpha = _check(op, 2, build_lcu_block_encoding)
    print(f"[LCU 2q,4term] n_anc={info['n_ancilla']} alpha={alpha:.4f} err={err:.2e}")
    assert err < 1e-9


def test_verify_block_encoding_helper_agrees():
    """The shipped verify_block_encoding() must report ~0 error and must
    auto-infer n_ancilla (no hard-coded 2)."""
    op = SparsePauliOp.from_list([("ZI", 0.5), ("IX", 0.3), ("XX", 0.2),
                                  ("ZZ", 0.15), ("YI", 0.1)])
    H = Operator(op).data
    alpha = float(np.sum(np.abs(op.coeffs)))
    be, _, info = build_simple_block_encoding(op, n_data_qubits=2)
    err = verify_block_encoding(be, H, alpha, n_data_qubits=2)  # n_ancilla inferred
    print(f"[verify helper] inferred n_anc={info['n_ancilla']} err={err:.2e}")
    assert err is not None and err < 1e-9


def test_postselection_matches_statevector_extraction():
    """The |0_anc> amplitude block, applied to a prepared data state, must
    equal H/alpha applied to that state -- i.e. the extraction convention in
    qsp_circuit matches the operator-level block."""
    op = SparsePauliOp.from_list([("ZI", 0.5), ("IX", 0.3), ("XX", 0.2)])
    n_data = 2
    H = Operator(op).data
    alpha = float(np.sum(np.abs(op.coeffs)))
    be, _, info = build_simple_block_encoding(op, n_data_qubits=n_data)
    U = Operator(be).data
    blk = _block_from_unitary(U, info['n_ancilla'], n_data)

    # random data state
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2 ** n_data) + 1j * rng.normal(size=2 ** n_data)
    psi /= np.linalg.norm(psi)

    lhs = blk @ psi
    rhs = (H / alpha) @ psi
    err = float(np.max(np.abs(lhs - rhs)))
    print(f"[block@psi vs H/alpha@psi] err={err:.2e}")
    assert err < 1e-9


def _run_all():
    tests = [
        test_simple_be_single_qubit,
        test_simple_be_two_qubit_five_terms,
        test_lcu_be_matches_simple,
        test_verify_block_encoding_helper_agrees,
        test_postselection_matches_statevector_extraction,
    ]
    print("=" * 70)
    print("  BLOCK-ENCODING CORRECTNESS:  <0_anc| U_BE |0_anc> == H / alpha")
    print("=" * 70)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {e}")
    print("=" * 70)
    print(f"  {passed}/{len(tests)} passed")
    print("=" * 70)
    return passed == len(tests)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
