"""
Smoke Test — Verify all imports and paths after restructure.
Run from project root:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/smoke_test.py
"""
import os
import sys
import traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

PASS = 0
FAIL = 0


def assert_(cond):
    if not cond:
        raise AssertionError("Assertion failed")


def check(label, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {label}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        FAIL += 1
        print()


print("=" * 60)
print("  SMOKE TEST — Post-restructure import & path checks")
print("=" * 60)

# --- 1. Core src imports ---
print("\n  --- src/ module imports ---")
check("data.gapdh_sequences",     lambda: __import__("data.gapdh_sequences"))
check("src.gy94_model",           lambda: __import__("src.gy94_model"))
check("src.hamiltonian",          lambda: __import__("src.hamiltonian"))
check("src.block_encoding",       lambda: __import__("src.block_encoding"))
check("src.aae_encoding",         lambda: __import__("src.aae_encoding"))
check("src.qsp_angles",           lambda: __import__("src.qsp_angles"))
check("src.qsp_circuit",          lambda: __import__("src.qsp_circuit"))
check("src.qsvt_angles_imagtime", lambda: __import__("src.qsvt_angles_imagtime"))
check("src.qsvt_circuit_imagtime",lambda: __import__("src.qsvt_circuit_imagtime"))
check("src.qsvt_imagtime_noisy",  lambda: __import__("src.qsvt_imagtime_noisy"))
check("src.trotter",              lambda: __import__("src.trotter"))
check("src.experiment",           lambda: __import__("src.experiment"))

# --- 2. Key function imports ---
print("\n  --- Key function imports ---")
check("build_gapdh_register", lambda: (
    from_import := __import__("data.gapdh_sequences", fromlist=["build_gapdh_register"]),
    getattr(from_import, "build_gapdh_register"),
))
check("build_gy94_rate_matrix", lambda: (
    m := __import__("src.gy94_model", fromlist=["build_gy94_rate_matrix"]),
    getattr(m, "build_gy94_rate_matrix"),
))
check("aae_encode", lambda: (
    m := __import__("src.aae_encoding", fromlist=["aae_encode"]),
    getattr(m, "aae_encode"),
))
check("compute_qsvt_angles_imagtime", lambda: (
    m := __import__("src.qsvt_angles_imagtime", fromlist=["compute_qsvt_angles_imagtime"]),
    getattr(m, "compute_qsvt_angles_imagtime"),
))
check("combine_imagtime_amplitudes", lambda: (
    m := __import__("src.qsvt_circuit_imagtime", fromlist=["combine_imagtime_amplitudes"]),
    getattr(m, "combine_imagtime_amplitudes"),
))

# --- 3. Path resolution checks ---
print("\n  --- Path resolution ---")
check("_PROJECT_DIR exists",      lambda: assert_(os.path.isdir(_PROJECT_DIR)))
check("data/ exists",             lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "data"))))
check("src/ exists",              lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "src"))))
check("scripts/ exists",          lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "scripts"))))
check("results/ exists",          lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "results"))))
check("figures/ exists",          lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "figures"))))
check("tests/ exists",            lambda: assert_(os.path.isdir(os.path.join(_PROJECT_DIR, "tests"))))

# --- 4. Quick pipeline sanity (lightweight, no training) ---
print("\n  --- Quick pipeline sanity ---")

def quick_pipeline():
    from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
    from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
    from src.hamiltonian import symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op
    from src.block_encoding import build_simple_block_encoding
    from src.trotter import classical_evolution
    import numpy as np

    codon_freqs = pooled_codon_frequencies()
    # GAPDH 4 species has 58 unique codons, not all 61 sense codons
    assert len(codon_freqs) >= 50, f"Expected ~58 codons, got {len(codon_freqs)}"

    best_v, min_err = 50.0, float('inf')
    for test_v in np.linspace(5, 200, 40):
        err = abs(calculate_implied_omega(codon_freqs, 1.8425, test_v) - 0.0599)
        if err < min_err:
            min_err, best_v = err, test_v

    Q, sense_codons, pi, _ = build_gy94_rate_matrix(codon_freqs, kappa=1.8425, V=best_v)
    assert Q.shape == (61, 61), f"Q shape {Q.shape}"

    H, _ = symmetrize_to_hamiltonian(Q, pi, n_qubits=6)
    assert H.shape == (64, 64), f"H shape {H.shape}"

    pauli_full, _ = decompose_to_pauli(H, n_qubits=6, threshold=1e-6)
    pauli_op, n_kept = filter_pauli_op(pauli_full, threshold=0.2)
    assert n_kept >= 1, f"No Pauli terms kept"

    be_circuit, alpha, be_info = build_simple_block_encoding(pauli_op, n_data_qubits=6)
    assert be_info['n_ancilla'] >= 1

    pi_cl, _ = classical_evolution(Q, pi, 0.5)
    assert abs(pi_cl.sum() - 1.0) < 1e-8

    print(f"    Codons={len(codon_freqs)}, Q={Q.shape}, H={H.shape}, "
          f"Pauli={n_kept} terms, alpha={alpha:.4f}, "
          f"BE ancilla={be_info['n_ancilla']}")

check("Full pipeline (Q->H->Pauli->BE->classical)", quick_pipeline)

# --- 5. Deprecated files should NOT be importable ---
print("\n  --- Deprecated stubs removed ---")
check("hamiltonian_qsvt NOT in src/", lambda: assert_(
    not os.path.exists(os.path.join(_PROJECT_DIR, "src", "hamiltonian_qsvt.py"))))
check("qsp_angles_qsvt NOT in src/", lambda: assert_(
    not os.path.exists(os.path.join(_PROJECT_DIR, "src", "qsp_angles_qsvt.py"))))
check("qsp_circuit_qsvt NOT in src/", lambda: assert_(
    not os.path.exists(os.path.join(_PROJECT_DIR, "src", "qsp_circuit_qsvt.py"))))
check("qsvt_results_gapdh NOT in root", lambda: assert_(
    not os.path.exists(os.path.join(_PROJECT_DIR, "qsvt_results_gapdh.py"))))

# --- Summary ---
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("  ALL CLEAR — safe to push!")
else:
    print("  FIX FAILURES BEFORE PUSHING")
print("=" * 60)
