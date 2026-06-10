"""
Centralized constants -- single source of truth
================================================
GY94 model parameters and pipeline defaults, calibrated to match the paper
exactly. Import these everywhere instead of hard-coding literals or running
ad-hoc grid searches that can drift between entry points.

Calibration (from PAML/codeml on the GAPDH 4-species alignment, Table 3):
    kappa = 1.8425   (transition/transversion ratio)
    omega = 0.0599   (dN/dS; strong purifying selection)
    V     = 13.5     (gene-specific variability; chosen so the Grantham-
                      augmented rate matrix reproduces dN/dS = 0.0599)

Rationale for fixing V instead of grid-searching it at runtime
--------------------------------------------------------------
PAML does not natively estimate V for the Grantham-augmented variant used
here. The paper adopts a two-step calibration: (kappa, omega) by maximum
likelihood under the standard model, then V by a one-time grid search such
that the augmented matrix reproduces the same global dN/dS = 0.0599. That
search yields V = 13.5. Re-running the search at import time (a) is slow,
(b) gave different answers at different entry points (391-point grid in the
QSVT main vs 40-point grid in the smoke test), and (c) made the Hamiltonian
non-deterministic. We therefore freeze the calibrated value here.

If you ever need to re-derive V (e.g. for a new gene/alignment), call
`calibrate_V()` below explicitly and then update GY94_V with the result.
"""

# --- GY94 calibrated parameters (paper Table 3) ---
GY94_KAPPA = 1.8425
GY94_OMEGA = 0.0599
GY94_V = 13.5

# --- Quantum register sizing ---
N_DATA_QUBITS = 6          # ceil(log2(61)) = 6 -> 64-dim Hilbert space
N_SENSE_CODONS = 61        # sense codons (64 - 3 stop)

# --- Pauli truncation thresholds studied in the paper (Tables 4/6/9) ---
PAULI_THRESHOLDS = (0.20, 0.10, 0.075, 0.05)
PAULI_THRESHOLD_OPTIMAL = 0.075     # peak Hellinger fidelity operating point
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


def calibrate_V(codon_frequencies, kappa=GY94_KAPPA, omega_target=GY94_OMEGA,
                v_grid=None):
    """
    One-time helper to derive V such that the Grantham-augmented GY94 matrix
    reproduces the target dN/dS. NOT called at runtime by the pipeline -- the
    calibrated result is frozen as GY94_V above. Provided for transparency
    and for re-calibration on new alignments.

    Parameters
    ----------
    codon_frequencies : dict   codon -> frequency
    kappa             : float
    omega_target      : float
    v_grid            : iterable of float or None
                        Defaults to a fine 391-point grid over [5, 200].

    Returns
    -------
    best_v   : float
    min_err  : float   |implied_omega(best_v) - omega_target|
    """
    import numpy as np
    from src.gy94_model import calculate_implied_omega

    if v_grid is None:
        v_grid = np.linspace(5.0, 200.0, 391)

    best_v, min_err = float(v_grid[0]), float("inf")
    for v in v_grid:
        err = abs(calculate_implied_omega(codon_frequencies, kappa, float(v)) - omega_target)
        if err < min_err:
            min_err, best_v = err, float(v)
    return best_v, min_err
