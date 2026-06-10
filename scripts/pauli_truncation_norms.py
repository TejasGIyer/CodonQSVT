"""
Pauli truncation error analysis
================================
Computes operator norm errors  ||H - H_tau||_2  (spectral)
                          and  ||H - H_tau||_F  (Frobenius)
for the four Pauli truncation thresholds (0.2, 0.1, 0.075, 0.05).

H_tau = sum_{|c_k| >= tau} c_k P_k    (the truncated Pauli operator)

Definitions
-----------
||M||_2 = largest singular value of M  (also = max |eigenvalue| if M is Hermitian)
||M||_F = sqrt( sum_{i,j} |M_ij|^2 )

Closed-form check
-----------------
Pauli strings on n qubits are orthogonal under Hilbert-Schmidt:
    tr(P_k^dagger P_l) = 2^n * delta_{kl}
so for H - H_tau = sum_{|c_k|<tau} c_k P_k (real c_k since H Hermitian):
    ||H - H_tau||_F^2 = 2^n * sum_{|c_k|<tau} c_k^2

We compute both and verify they match (sanity check on the SparsePauliOp).

Place this file in:  C:\\Users\\Ganesh\\gene_mutation_main\\scripts\\
Run from project root:
    python scripts/pauli_truncation_norms.py
"""

import os
import sys
import json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qiskit.quantum_info import SparsePauliOp

from data.gapdh_sequences import pooled_codon_frequencies
from src.gy94_model import build_gy94_rate_matrix, calculate_implied_omega
from src.hamiltonian import (
    symmetrize_to_hamiltonian, decompose_to_pauli, filter_pauli_op,
)


# =====================================================================
# CONFIG — match the rest of the pipeline
# =====================================================================
KAPPA      = 1.8425
OMEGA      = 0.0599
N_QUBITS   = 6
THRESHOLDS = [0.2, 0.1, 0.075, 0.05]   # the four reported in QSVT_RESULTS

# Threshold for the underlying "full" decomposition (numerical floor).
# Anything dropped here is below machine-level relevance.
FLOOR_THRESHOLD = 1e-8


# =====================================================================
# Helpers
# =====================================================================
def H_from_pauli_op(pauli_op):
    """Reconstruct the dense Hermitian matrix from a SparsePauliOp.
    Real-cast to discard sub-1e-12 imaginary roundoff."""
    M = pauli_op.to_matrix()
    if np.max(np.abs(M.imag)) > 1e-10:
        raise ValueError(
            f"Pauli operator is not Hermitian to 1e-10 "
            f"(max imag = {np.max(np.abs(M.imag)):.2e})")
    return M.real


def spectral_norm(M):
    """||M||_2 = largest singular value. For Hermitian M this equals
    max |eigenvalue|, which is faster — use eigvalsh."""
    if np.allclose(M, M.T, atol=1e-12):
        return float(np.max(np.abs(np.linalg.eigvalsh(M))))
    return float(np.linalg.norm(M, ord=2))


def frobenius_norm(M):
    return float(np.linalg.norm(M, ord='fro'))


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 78)
    print("  Pauli truncation norm error analysis")
    print("=" * 78)

    # ---------- Build H exactly as the rest of the pipeline does ----------
    print("\n[1/3] Building GY94 + symmetrizing to H...")
    codon_freqs = pooled_codon_frequencies()
    best_v, min_err = 50.0, float('inf')
    for test_v in np.linspace(5, 200, 391):
        err = abs(calculate_implied_omega(codon_freqs, KAPPA, test_v) - OMEGA)
        if err < min_err:
            min_err, best_v = err, test_v
    print(f"   V = {best_v:.4f}  (omega err = {min_err:.2e})")

    Q, sense_codons, pi, _ = build_gy94_rate_matrix(
        codon_freqs, kappa=KAPPA, V=best_v)
    H, h_info = symmetrize_to_hamiltonian(Q, pi, n_qubits=N_QUBITS)
    print(f"   H shape = {H.shape},  ||H||_F = {h_info['frobenius_norm']:.6f}")
    print(f"   ||H||_2 = {h_info['spectral_norm']:.6f}  (spectral norm)")

    # ---------- Reference full Pauli decomposition ----------
    print(f"\n[2/3] Full Pauli decomposition (floor threshold = {FLOOR_THRESHOLD:.0e})...")
    pauli_full, p_info = decompose_to_pauli(
        H, n_qubits=N_QUBITS, threshold=FLOOR_THRESHOLD)
    print(f"   {p_info['n_significant_terms']} Pauli terms kept (of {p_info['n_total_pauli_terms']} total)")
    print(f"   sum |c_k| (1-norm = alpha at this floor) = {p_info['total_pauli_norm']:.6f}")

    # Sanity check: reconstructing H from the full Pauli op should reproduce H
    H_reconstructed = H_from_pauli_op(pauli_full)
    recon_err_F = frobenius_norm(H - H_reconstructed)
    recon_err_2 = spectral_norm(H - H_reconstructed)
    print(f"   Reconstruction error ||H - H_full||_F = {recon_err_F:.3e}")
    print(f"   Reconstruction error ||H - H_full||_2 = {recon_err_2:.3e}")
    if recon_err_F > 1e-6:
        print("   WARNING: large reconstruction error; results below treat")
        print("            H_full as the reference, not the original H.")

    # ---------- Per-threshold truncation errors ----------
    print(f"\n[3/3] Computing truncation errors for each threshold...")
    print()

    # Use the FULL Pauli op (above the floor) as the reference matrix.
    # Difference operators are exactly the Paulis with |c_k| in [floor, tau).
    full_coeffs = np.real(pauli_full.coeffs).astype(float)
    full_labels = pauli_full.paulis.to_labels()
    dim = 2 ** N_QUBITS    # = 64

    rows = []
    header = (f"  {'threshold':>10}  {'n_terms':>8}  {'n_dropped':>10}  "
              f"{'alpha':>9}  {'||H-H_t||_2':>13}  {'||H-H_t||_F':>13}  "
              f"{'||H-H_t||_F (closed)':>22}  {'rel_F':>9}  {'rel_2':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    H_ref = H_from_pauli_op(pauli_full)        # exact, dense H
    H_ref_F = frobenius_norm(H_ref)
    H_ref_2 = spectral_norm(H_ref)

    for tau in THRESHOLDS:
        # Kept terms
        keep_mask = np.abs(full_coeffs) >= tau
        kept_labels = [full_labels[i] for i in range(len(full_labels)) if keep_mask[i]]
        kept_coeffs = full_coeffs[keep_mask]
        n_kept = int(keep_mask.sum())
        n_dropped = int((~keep_mask).sum())
        alpha_tau = float(np.sum(np.abs(kept_coeffs)))

        # H_tau (truncated)
        pauli_tau = SparsePauliOp.from_list(list(zip(kept_labels, kept_coeffs)))
        H_tau = H_from_pauli_op(pauli_tau)

        # Errors
        diff = H_ref - H_tau
        err_2 = spectral_norm(diff)
        err_F = frobenius_norm(diff)

        # Closed-form Frobenius (Pauli orthogonality):
        #   ||sum c_k P_k||_F^2 = 2^n * sum c_k^2
        dropped_coeffs = full_coeffs[~keep_mask]
        err_F_closed = float(np.sqrt(dim * np.sum(dropped_coeffs ** 2)))

        rel_F = err_F / H_ref_F if H_ref_F > 0 else float('nan')
        rel_2 = err_2 / H_ref_2 if H_ref_2 > 0 else float('nan')

        print(f"  {tau:>10.3f}  {n_kept:>8d}  {n_dropped:>10d}  "
              f"{alpha_tau:>9.4f}  {err_2:>13.6f}  {err_F:>13.6f}  "
              f"{err_F_closed:>22.6f}  {rel_F:>9.4f}  {rel_2:>9.4f}")

        rows.append({
            'threshold': tau,
            'n_pauli_terms': n_kept,
            'n_dropped':     n_dropped,
            'alpha':         alpha_tau,
            'spectral_norm_error':  err_2,
            'frobenius_norm_error': err_F,
            'frobenius_norm_error_closed_form': err_F_closed,
            'relative_frobenius_error': rel_F,
            'relative_spectral_error': rel_2,
        })

    print()
    print(f"  Reference  ||H||_2 = {H_ref_2:.6f}")
    print(f"  Reference  ||H||_F = {H_ref_F:.6f}")

    # ---------- Save ----------
    out = {
        'config': {
            'kappa':    KAPPA,
            'omega':    OMEGA,
            'V':        best_v,
            'n_qubits': N_QUBITS,
            'floor_threshold': FLOOR_THRESHOLD,
            'H_norm_2': H_ref_2,
            'H_norm_F': H_ref_F,
        },
        'rows': rows,
    }
    out_path = os.path.join(_PROJECT_DIR, 'results', 'pauli_truncation_norms.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    # ---------- Pretty LaTeX-style table for paper inclusion ----------
    print()
    print("=" * 78)
    print("  LaTeX-ready table (copy into your paper):")
    print("=" * 78)
    print()
    print(r"\begin{tabular}{c c c c c}")
    print(r"\hline")
    print(r"$\tau$ & $N_\mathrm{Pauli}$ & $\alpha$ & $\|H-H_\tau\|_2$ & $\|H-H_\tau\|_F$ \\")
    print(r"\hline")
    for r in rows:
        print(f"  {r['threshold']:.3f} & {r['n_pauli_terms']} & "
              f"{r['alpha']:.4f} & {r['spectral_norm_error']:.4f} & "
              f"{r['frobenius_norm_error']:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print()


if __name__ == "__main__":
    main()
