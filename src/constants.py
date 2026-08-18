"""
Centralized constants -- single source of truth
================================================
GY94 model parameters and pipeline defaults, calibrated to match the paper
exactly. Import these everywhere instead of hard-coding literals or running
ad-hoc grid searches that can drift between entry points.

Calibration (from PAML/codeml on the GAPDH 4-species alignment, Table 3):
    kappa = 1.8425   (transition/transversion ratio)
    omega = 0.0599   (dN/dS; strong purifying selection)
    V     = 13.6853  (gene-specific variability; exact root chosen so the
                      Grantham-augmented rate matrix reproduces dN/dS = 0.0599)

Rationale for fixing V instead of grid-searching it at runtime
--------------------------------------------------------------
PAML does not natively estimate V for the Grantham-augmented variant used
here. The paper adopts a two-step calibration: (kappa, omega) by maximum
likelihood under the standard model, then V by one-time root finding such
that the augmented matrix reproduces the same global dN/dS = 0.0599. Exact
root finding yields V = 13.6852722300. Re-running calibration at import time is
slow,
(b) gave different answers at different entry points (391-point grid in the
QSVT main vs 40-point grid in the smoke test), and (c) made the Hamiltonian
non-deterministic. We therefore freeze the calibrated value here.

If you ever need to re-derive V (e.g. for a new gene/alignment), call
`calibrate_V()` below explicitly and then update GY94_V with the result.
"""

# --- GY94 calibrated parameters (paper Table 3) ---
GY94_KAPPA = 1.8425
GY94_OMEGA = 0.0599
# V calibrated by exact root-finding (scipy.optimize.brentq), not a grid.
# brentq gives V = 13.6852722300 with implied omega = 0.0599000000
# (residual 1.1e-16). The previous grid value 13.5 gave omega = 0.058606,
# i.e. 2.2% below the PAML target -- a grid artifact, not a calibration.
GY94_V = 13.6853

# --- Quantum register sizing ---
N_DATA_QUBITS = 6          # ceil(log2(61)) = 6 -> 64-dim Hilbert space
N_SENSE_CODONS = 61        # sense codons (64 - 3 stop)

# --- Pauli truncation thresholds studied in the paper (Tables 4/6/9) ---
PAULI_THRESHOLDS = (0.20, 0.10, 0.075, 0.05)
PAULI_THRESHOLD_OPTIMAL = 0.20      # only validated imaginary-time operating point
PAULI_THRESHOLD_PRIMARY = 0.20      # shallowest / primary statevector demo

# --- Full-decomposition threshold (effectively "keep everything") ---
PAULI_FULL_THRESHOLD = 1e-6

# --- Default evolution time used across demos ---
T_EVOL_DEFAULT = 0.5

# --- AAE training reproducibility ---
AAE_N_LAYERS = 8
AAE_N_TRIALS = 6
AAE_RANDOM_SEED = 1234     # seed so "best of n_trials" is reproducible

# --- Dataset tag for cached artifacts ---
DATASET_TAG = "GAPDH_4species"
AAE_CACHE_FILENAME = "best_aae_params_gapdh_probability.json"


def calibrate_V(codon_frequencies, kappa=GY94_KAPPA, omega_target=GY94_OMEGA,
                bracket=(5.0, 200.0)):
    """
    Derive V such that the Grantham-augmented GY94 matrix reproduces the
    target dN/dS, by exact root-finding rather than grid search.

    implied_omega(V) is smooth and monotone increasing in V, so brentq
    converges to machine precision. The old grid approach produced different
    answers at different call sites (391-point grid -> 13.5, 40-point grid in
    the smoke test -> 15.0), which silently changed the Hamiltonian.

    Returns
    -------
    best_v  : float
    err     : float   |implied_omega(best_v) - omega_target|
    """
    from scipy.optimize import brentq
    from src.gy94_model import calculate_implied_omega

    f = lambda v: calculate_implied_omega(codon_frequencies, kappa, float(v)) - omega_target
    best_v = float(brentq(f, bracket[0], bracket[1], xtol=1e-10, rtol=1e-12))
    err = abs(calculate_implied_omega(codon_frequencies, kappa, best_v) - omega_target)
    return best_v, err


# --- Thresholds that preserve negative-semidefiniteness of H_tau ---
# Truncation can push eigenvalues positive, which breaks the dissipativity
# guarantee that e^{H t} is a contraction. Measured lam_max(H_tau):
#     tau=0.20  ->  -0.1417   (dissipative)
#     tau=0.10  ->  +0.4318   (amplifies)
#     tau=0.075 ->  +0.4501   (amplifies)
#     tau=0.05  ->  +0.2807   (amplifies)
# Only 0.20 is compatible with the global odd-channel sign correction used by
# the imaginary-time implementation. Use assert_dissipative() below before
# making any statement about monotone norm decay at a given threshold.
PAULI_THRESHOLDS_DISSIPATIVE = (0.20,)


def assert_dissipative(pauli_op, tol=1e-10, raise_on_fail=False):
    """
    Check that the truncated generator is still negative-semidefinite.

    Returns (is_dissipative, lam_max). Call this in any sweep that claims
    norm decay, so a positive eigenvalue is reported rather than silently
    producing a growing norm.
    """
    import numpy as np
    from qiskit.quantum_info import Operator
    lam_max = float(np.max(np.linalg.eigvalsh(np.real(Operator(pauli_op).data))))
    ok = lam_max <= tol
    if not ok and raise_on_fail:
        raise ValueError(
            f"Truncated H_tau is NOT negative-semidefinite: lam_max = {lam_max:+.4f}. "
            f"e^(H_tau t) will amplify along that mode; the norm-decay claim "
            f"does not hold at this threshold.")
    return ok, lam_max
