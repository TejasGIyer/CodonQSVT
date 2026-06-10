"""
Step 4-5: Hamiltonian Construction & Pauli Decomposition
=========================================================
Converts the GY94 rate matrix Q into a Hermitian Hamiltonian H
suitable for quantum simulation (QSVT imaginary-time, or Trotter baseline).

Step 4: Symmetrize Q -> H using detailed balance
    H = D^{1/2} Q D^{-1/2}
    where D = diag(pi_1, pi_2, ..., pi_n)

    Because GY94 satisfies detailed balance (pi_i Q_ij = pi_j Q_ji),
    H is guaranteed to be symmetric (Hermitian). This is the ONLY embedding
    used in this project; the antidiagonal dilation [[0,Q],[Q^T,0]] that once
    lived in the (now-removed) *_qsvt test modules is NOT used anywhere.

Step 5: Decompose H into Pauli strings
    H = Sum_k c_k P_k
    where P_k are tensor products of Pauli matrices (I, X, Y, Z)
    and c_k are real coefficients.

Reference: Goldman & Yang (1994) confirm reversibility of Q.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator


# =========================================================================
# PAULI FILTER HELPER
# =========================================================================

def filter_pauli_op(pauli_op_full, threshold):
    """
    Re-filter an already-decomposed SparsePauliOp at a new threshold.
    Keeps only terms with |coefficient| >= threshold.

    Parameters
    ----------
    pauli_op_full : SparsePauliOp  (full decomposition, e.g. threshold=1e-6)
    threshold     : float          (new, higher threshold to apply)

    Returns
    -------
    filtered_op   : SparsePauliOp
    n_kept        : int
    """
    coeffs = pauli_op_full.coeffs
    labels = pauli_op_full.paulis.to_labels()

    mask   = np.abs(coeffs) >= threshold
    n_kept = int(np.sum(mask))

    kept_labels = [labels[i] for i in range(len(labels)) if mask[i]]
    kept_coeffs = coeffs[mask]

    filtered_op = SparsePauliOp.from_list(
        [(l, c) for l, c in zip(kept_labels, kept_coeffs)]
    )
    return filtered_op, n_kept


def symmetrize_to_hamiltonian(Q, pi, n_qubits=6, pad_eigenvalue='lam_min'):
    """
    Step 4: Symmetrize the GY94 rate matrix Q into a Hermitian Hamiltonian H.

    The transformation H = D^{1/2} Q D^{-1/2} exploits detailed balance
    (pi_i Q_ij = pi_j Q_ji) to produce a symmetric matrix.

    Spectral-padding fix
    --------------------
    The 58 observed (of 61 sense) codons are embedded into a 2^n_qubits
    register. Padding the unused basis states with ZEROS makes them spurious
    zero eigenvalues of H, which would contradict the paper's claim of a
    SINGLE zero eigenvalue (the stationary distribution). To avoid this we
    place every padding / unobserved basis state on a DECOUPLED diagonal
    entry equal to the most negative physical eigenvalue lam_min (default).

    Because the padding block is decoupled (all off-diagonals zero), these
    states never exchange amplitude with the physical block under e^{Ht};
    they simply sit at the edge of the existing spectral window [lam_min, 0]
    instead of introducing new zero modes. The physical spectrum is preserved
    exactly and the spectral norm (hence the H/alpha in [-1, 0] requirement)
    is unchanged.

    Parameters
    ----------
    Q  : (61x61) GY94 rate matrix
    pi : (61,)   equilibrium codon frequencies
    n_qubits : int   number of qubits (default 6 -> 64 states)
    pad_eigenvalue : 'lam_min' | float | None
        'lam_min' (default) places padding states at the most negative
        physical eigenvalue (no new zero modes, window unchanged);
        a float places them at that explicit value;
        None reverts to legacy zero-padding (NOT recommended -- reintroduces
        spurious zero eigenvalues).

    Returns
    -------
    H    : (2^n x 2^n) Hermitian Hamiltonian matrix (padded)
    info : dict with diagnostic information
    """
    n_sense = len(pi)            # 61
    n_states = 2 ** n_qubits     # 64 for n_qubits=6

    # Build D^{1/2} and D^{-1/2} for sense codons
    sqrt_pi = np.zeros(n_sense)
    inv_sqrt_pi = np.zeros(n_sense)
    for i in range(n_sense):
        if pi[i] > 0:
            sqrt_pi[i] = np.sqrt(pi[i])
            inv_sqrt_pi[i] = 1.0 / np.sqrt(pi[i])

    # Symmetrize: H_ij = sqrt(pi_i) * Q_ij * (1/sqrt(pi_j))
    H_small = np.zeros((n_sense, n_sense))
    for i in range(n_sense):
        for j in range(n_sense):
            H_small[i, j] = sqrt_pi[i] * Q[i, j] * inv_sqrt_pi[j] if (pi[i] > 0 and pi[j] > 0) else 0.0

    # Verify symmetry, then enforce it exactly (remove numerical noise)
    symmetry_error = np.max(np.abs(H_small - H_small.T))
    H_small = (H_small + H_small.T) / 2.0

    # Which physical rows are populated (observed codons)? Unobserved sense
    # codons (pi == 0) yield all-zero rows/cols -> spurious zero eigenvalues.
    observed = np.where(pi > 0)[0]
    n_observed = int(len(observed))

    # Most negative physical eigenvalue, computed over the observed support.
    if n_observed > 0:
        phys_eigs = np.linalg.eigvalsh(H_small[np.ix_(observed, observed)])
        lam_min = float(np.min(phys_eigs))
    else:
        lam_min = 0.0

    # Pad to 2^n_qubits x 2^n_qubits
    H = np.zeros((n_states, n_states))
    H[:n_sense, :n_sense] = H_small

    # Resolve the padding diagonal value.
    if pad_eigenvalue is None:
        pad_val = 0.0                       # legacy (discouraged)
    elif pad_eigenvalue == 'lam_min':
        pad_val = lam_min
    else:
        pad_val = float(pad_eigenvalue)

    # Padding / unused basis states: trailing states beyond n_sense, PLUS any
    # unobserved sense codon (pi == 0). Decoupled diagonal at pad_val.
    if pad_val != 0.0:
        unused = list(range(n_sense, n_states))
        unused += [int(i) for i in range(n_sense) if pi[i] <= 0]
        for k in unused:
            H[k, k] = pad_val

    # Hermitian check + spectrum
    is_hermitian = np.allclose(H, H.T, atol=1e-12)
    eigenvalues = np.real(np.linalg.eigvalsh(H))
    eigenvalues_sorted = np.sort(eigenvalues)

    spectral_norm = np.max(np.abs(eigenvalues))
    spectral_gap = (eigenvalues_sorted[-1] - eigenvalues_sorted[-2]
                    if len(eigenvalues_sorted) > 1 else 0)

    info = {
        'n_sense': n_sense,
        'n_observed': n_observed,
        'n_qubits': n_qubits,
        'n_states': n_states,
        'pad_eigenvalue': pad_val,
        'lam_min': lam_min,
        'symmetry_error_before_enforce': symmetry_error,
        'is_hermitian': is_hermitian,
        'eigenvalue_min': eigenvalues_sorted[0],
        'eigenvalue_max': eigenvalues_sorted[-1],
        'spectral_norm': spectral_norm,
        'spectral_gap': spectral_gap,
        'n_zero_eigenvalues': int(np.sum(np.abs(eigenvalues) < 1e-10)),
        'frobenius_norm': np.linalg.norm(H, 'fro'),
    }

    return H, info


def decompose_to_pauli(H, n_qubits=6, threshold=1e-8):
    """
    Step 5: Decompose the Hamiltonian H into a sum of Pauli strings.

    H = Sum_k c_k P_k

    Parameters
    ----------
    H : (2^n x 2^n) Hermitian matrix
    n_qubits : int
    threshold : float   drop Pauli terms with |coefficient| < threshold

    Returns
    -------
    pauli_op : SparsePauliOp representing H
    info     : dict with diagnostic information
    """
    operator = Operator(H)
    pauli_op_full = SparsePauliOp.from_operator(operator)

    coeffs = pauli_op_full.coeffs
    labels = pauli_op_full.paulis.to_labels()

    mask = np.abs(coeffs) >= threshold
    n_total = len(coeffs)
    n_kept = int(np.sum(mask))

    significant_labels = [labels[i] for i in range(n_total) if mask[i]]
    significant_coeffs = coeffs[mask]

    pauli_op = SparsePauliOp.from_list(
        [(label, coeff) for label, coeff in zip(significant_labels, significant_coeffs)]
    )

    max_imag = np.max(np.abs(np.imag(significant_coeffs))) if len(significant_coeffs) else 0.0

    weight_dist = {}
    for label in significant_labels:
        weight = sum(1 for c in label if c != 'I')
        weight_dist[weight] = weight_dist.get(weight, 0) + 1

    sorted_indices = np.argsort(-np.abs(significant_coeffs))
    top_terms = [(significant_labels[i], float(np.real(significant_coeffs[i])))
                 for i in sorted_indices[:10]]

    info = {
        'n_total_pauli_terms': n_total,
        'n_significant_terms': n_kept,
        'n_dropped_terms': n_total - n_kept,
        'threshold': threshold,
        'max_imaginary_part': max_imag,
        'weight_distribution': weight_dist,
        'top_10_terms': top_terms,
        'total_pauli_norm': float(np.sum(np.abs(significant_coeffs))),
    }

    return pauli_op, info


def print_hamiltonian_report(H, h_info, pauli_op, p_info):
    """Print a detailed report of the Hamiltonian and its Pauli decomposition."""
    print("\n" + "=" * 70)
    print("  STEP 4: HAMILTONIAN CONSTRUCTION (SYMMETRIZATION)")
    print("=" * 70)

    print(f"\n  Transformation: H = D^(1/2) x Q x D^(-1/2)")
    print(f"  Matrix size: {h_info['n_sense']}x{h_info['n_sense']} -> padded to {h_info['n_states']}x{h_info['n_states']}")
    print(f"  Qubits: {h_info['n_qubits']}")
    print(f"  Observed codons: {h_info.get('n_observed', 'n/a')}")

    print(f"\n  Symmetry verification:")
    print(f"    Symmetry error (before enforce): {h_info['symmetry_error_before_enforce']:.2e}")
    print(f"    Is Hermitian (after enforce):    {h_info['is_hermitian']}")

    print(f"\n  Spectral properties:")
    print(f"    Eigenvalue range: [{h_info['eigenvalue_min']:.6f}, {h_info['eigenvalue_max']:.6f}]")
    print(f"    Spectral norm:    {h_info['spectral_norm']:.6f}")
    print(f"    Spectral gap:     {h_info['spectral_gap']:.6f}")
    print(f"    Padding eigenvalue: {h_info.get('pad_eigenvalue', 0.0):.6f}  (lam_min={h_info.get('lam_min', 0.0):.6f})")
    print(f"    Zero eigenvalues: {h_info['n_zero_eigenvalues']}  (should be 1: the stationary mode)")
    print(f"    Frobenius norm:   {h_info['frobenius_norm']:.6f}")

    print(f"\n" + "=" * 70)
    print(f"  STEP 5: PAULI DECOMPOSITION")
    print(f"=" * 70)

    print(f"\n  H = Sum_k c_k P_k")
    print(f"    Total Pauli terms:       {p_info['n_total_pauli_terms']}")
    print(f"    Significant terms:       {p_info['n_significant_terms']} (|c_k| >= {p_info['threshold']})")
    print(f"    Dropped terms:           {p_info['n_dropped_terms']}")
    print(f"    Max imaginary part:      {p_info['max_imaginary_part']:.2e} (should be ~0)")
    print(f"    Total Pauli norm:        {p_info['total_pauli_norm']:.6f}")

    print(f"\n  Pauli weight distribution (weight = number of non-I operators):")
    for w in sorted(p_info['weight_distribution']):
        print(f"    Weight {w}: {p_info['weight_distribution'][w]} terms")

    print(f"\n  Top 10 Pauli terms by magnitude:")
    print(f"  {'Rank':>4}  {'Pauli String':>12}  {'Coefficient':>12}  {'Weight':>6}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*6}")
    for i, (label, coeff) in enumerate(p_info['top_10_terms']):
        weight = sum(1 for c in label if c != 'I')
        print(f"  {i+1:4d}  {label:>12}  {coeff:12.8f}  {weight:6d}")


# =========================================================================
# STANDALONE TEST -- removed. Use src/qsp_circuit.py as the entry point.
# =========================================================================
