"""
QSVT Phase Angles for Imaginary-Time Evolution (e^{Ht})
========================================================
Computes cosh and sinh phase sequences that, when combined, compute the
NON-UNITARY imaginary-time evolution e^{Ht} for negative-semi-definite H.

This replaces the QSP cos/sin angles (which compute UNITARY e^{-iHt}).
The circuit topology is IDENTICAL to the cos/sin pipeline — only the phase
angles and the classical combination at the end change.

Math
----
After symmetrization H = D^{1/2} Q D^{-1/2}, H has the same eigenvalues
as Q: all real, all <= 0. After block-encoding by alpha = sum|c_k|,
the spectrum of H/alpha lies in [-1, 0].

We want e^{Ht} = e^{(H/alpha)*tau} where tau = alpha * t.
We split into definite parity:

    e^{tau*x} = cosh(tau*x) + sinh(tau*x)
                  (even)        (odd)

QSVT representability requires |P(x)| <= 1 on [-1, 1], so we rescale:

    target_cosh(x) = cosh(tau*x) / (2 * cosh(tau))   <= 1/2
    target_sinh(x) = sinh(tau*x) / (2 * cosh(tau))   <= 1/2

After running BOTH channels and combining:
    e^{tau*x} = (target_cosh + target_sinh) * 2 * cosh(tau)

Post-selection probability decays with t — this is the quantum signature
of classical thermalization.

Reference
---------
Gilyen, Su, Low, Wiebe (2019), arXiv:1806.01838 (sec. 3).
"""

import numpy as np
import numpy.polynomial.chebyshev as cheb
from pyqsp.angle_sequence import QuantumSignalProcessingPhases


# =====================================================================
# CHEBYSHEV POLYNOMIAL FITTING
# =====================================================================

def chebyshev_coefs_from_function(func, degree, parity):
    """
    Compute Chebyshev expansion coefficients of func(x) on [-1, 1],
    enforcing definite parity (even or odd).

    Uses numpy.polynomial.chebyshev.Chebyshev.interpolate for numerical
    stability, then zeros out wrong-parity coefficients.
    """
    chebpoly = cheb.Chebyshev.interpolate(func, degree, domain=[-1, 1])
    coefs = chebpoly.coef.copy()
    if parity == 'even':
        coefs[1::2] = 0.0
    elif parity == 'odd':
        coefs[0::2] = 0.0
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    return coefs


def estimate_chebyshev_degree(tau, epsilon, parity):
    """
    Heuristic for the polynomial degree needed to approximate
    cosh/sinh(tau*x) to within epsilon in sup-norm.
    """
    base = int(np.ceil(1.5 * tau + np.log2(max(1.0 / epsilon, 2.0))))
    deg = max(base, 5)
    if parity == 'even' and deg % 2 == 1:
        deg += 1
    elif parity == 'odd' and deg % 2 == 0:
        deg += 1
    return deg


# =====================================================================
# MAIN ANGLE COMPUTATION
# =====================================================================

def compute_qsvt_angles_imagtime(alpha, t, epsilon=1e-3,
                                 degree_cosh=None, degree_sinh=None):
    """
    Compute QSVT phase angles for imaginary-time evolution e^{Ht}.

    Parameters
    ----------
    alpha : float
        1-norm of the Pauli decomposition (block-encoding normalization).
    t : float
        Evolution time.
    epsilon : float
        Target sup-norm error of the polynomial approximation.
    degree_cosh, degree_sinh : int or None
        Optional manual override of polynomial degree per channel.

    Returns
    -------
    phases_cosh : np.ndarray   QSP phases for cosh channel (EVEN parity).
    phases_sinh : np.ndarray   QSP phases for sinh channel (ODD parity).
    info : dict
        norm_factor_cosh, norm_factor_sinh = multipliers to undo rescaling.
        norm_factor_base = 2*cosh(tau).
    """
    tau = alpha * t
    norm_factor = 2.0 * np.cosh(tau)

    # Target functions rescaled to |P(x)| <= 1 on [-1, 1]
    f_cosh = lambda x: np.cosh(tau * x) / norm_factor
    f_sinh = lambda x: np.sinh(tau * x) / norm_factor

    # Choose polynomial degrees
    if degree_cosh is None:
        degree_cosh = estimate_chebyshev_degree(tau, epsilon, 'even')
    if degree_sinh is None:
        degree_sinh = estimate_chebyshev_degree(tau, epsilon, 'odd')

    # Fit Chebyshev coefficients
    coefs_cosh = chebyshev_coefs_from_function(f_cosh, degree_cosh, parity='even')
    coefs_sinh = chebyshev_coefs_from_function(f_sinh, degree_sinh, parity='odd')

    # Validate bound: max |P(x)| on [-1, 1] should be <= 1
    x_test = np.linspace(-1.0, 1.0, 2001)
    p_cosh_vals = cheb.chebval(x_test, coefs_cosh)
    p_sinh_vals = cheb.chebval(x_test, coefs_sinh)
    max_cosh = float(np.max(np.abs(p_cosh_vals)))
    max_sinh = float(np.max(np.abs(p_sinh_vals)))

    # Safety clamp if any value exceeds 1
    safety = 1.0 / 1.02  # 2% margin
    if max_cosh > 1.0:
        coefs_cosh = coefs_cosh * (safety / max_cosh)
        norm_factor_cosh = norm_factor * (max_cosh / safety)
    else:
        norm_factor_cosh = norm_factor
    if max_sinh > 1.0:
        coefs_sinh = coefs_sinh * (safety / max_sinh)
        norm_factor_sinh = norm_factor * (max_sinh / safety)
    else:
        norm_factor_sinh = norm_factor

    # Compute QSP phases via pyqsp Laurent method
    phases_cosh = np.array(QuantumSignalProcessingPhases(
        coefs_cosh.tolist(), signal_operator='Wx', method='laurent'))
    phases_sinh = np.array(QuantumSignalProcessingPhases(
        coefs_sinh.tolist(), signal_operator='Wx', method='laurent'))

    # Diagnostics: measure actual approx error on [-1, 0] (the physical domain)
    x_neg = np.linspace(-1.0, 0.0, 1001)
    true_exp = np.exp(tau * x_neg)
    approx = (cheb.chebval(x_neg, coefs_cosh) * norm_factor_cosh
              + cheb.chebval(x_neg, coefs_sinh) * norm_factor_sinh)
    approx_error = float(np.max(np.abs(true_exp - approx)))

    info = {
        'tau': tau,
        'alpha': alpha,
        't': t,
        'epsilon': epsilon,
        'norm_factor_cosh': float(norm_factor_cosh),
        'norm_factor_sinh': float(norm_factor_sinh),
        'norm_factor_base': float(norm_factor),
        'cosh_degree': int(degree_cosh),
        'sinh_degree': int(degree_sinh),
        'n_cosh_phases': int(len(phases_cosh)),
        'n_sinh_phases': int(len(phases_sinh)),
        'cosh_max_poly_value': max_cosh,
        'sinh_max_poly_value': max_sinh,
        'approx_error_on_neg_interval': approx_error,
        'expected_postselection_decay_at_x_minus_1': float(np.exp(-tau)),
    }

    return phases_cosh, phases_sinh, info


def print_qsvt_angles_report(phases_cosh, phases_sinh, info):
    print("\n" + "=" * 70)
    print("  QSVT PHASE ANGLES  (imaginary-time evolution e^{Ht})")
    print("=" * 70)
    print(f"\n  Parameters:")
    print(f"    alpha (1-norm):           {info['alpha']:.4f}")
    print(f"    t (evolution time):       {info['t']:.4f}")
    print(f"    tau = alpha * t:          {info['tau']:.4f}")
    print(f"    epsilon (target):         {info['epsilon']:.2e}")
    print(f"\n  Normalization:")
    print(f"    norm_factor (= 2 cosh tau):  {info['norm_factor_base']:.4f}")
    print(f"    cosh poly max on [-1,1]:     {info['cosh_max_poly_value']:.4f}")
    print(f"    sinh poly max on [-1,1]:     {info['sinh_max_poly_value']:.4f}")
    print(f"\n  Cosh channel (even parity):")
    print(f"    Polynomial degree:    {info['cosh_degree']}")
    print(f"    QSP phases:           {info['n_cosh_phases']}")
    print(f"\n  Sinh channel (odd parity):")
    print(f"    Polynomial degree:    {info['sinh_degree']}")
    print(f"    QSP phases:           {info['n_sinh_phases']}")
    print(f"\n  Approximation quality:")
    print(f"    max |e^(tau*x) - poly|  on x in [-1,0]:  {info['approx_error_on_neg_interval']:.3e}")
    print(f"\n  Expected post-selection decay at x=-1:")
    print(f"    {info['expected_postselection_decay_at_x_minus_1']:.4f}")


if __name__ == "__main__":
    print("Computing QSVT imaginary-time phases for alpha=1.7, t=0.5...")
    pc, ps, info = compute_qsvt_angles_imagtime(alpha=1.6977, t=0.5, epsilon=1e-3)
    print_qsvt_angles_report(pc, ps, info)
