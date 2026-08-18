"""
Readout / reference-convention correctness.

Two invariants that must hold exactly, independent of any quantum circuit:

  1. classical_evolution must fix the stationary distribution:
         P_t^T pi_eq == pi_eq  for all t.
  2. reweight_to_distribution must be the exact inverse of the symmetrization:
         feeding it the probabilities implied by the true pi(t) must return
         pi(t) with F_H = 1.

Both were violated before: (1) drifted by 0.389 in L1 at t=0.5, and (2)
returned a constant F_H = 0.894.

Run:  pytest tests/test_readout_inversion.py -v
"""
import os
import sys
import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TEST_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from data.gapdh_sequences import build_gapdh_register, pooled_codon_frequencies
from src.gy94_model import build_gy94_rate_matrix
from src.hamiltonian import reweight_to_distribution
from src.trotter import classical_evolution


def _hellinger(p, q):
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    if p.sum() > 1e-12: p = p / p.sum()
    if q.sum() > 1e-12: q = q / q.sum()
    return float(np.clip(1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2), 0.0, 1.0))


def _setup():
    Q, sense_codons, pi, _ = build_gy94_rate_matrix(pooled_codon_frequencies())
    return Q, pi


def test_stationary_distribution_is_fixed():
    """pi_eq must be invariant under the classical propagator, for every t."""
    Q, pi = _setup()
    for t in (0.1, 0.5, 1.0, 5.0):
        pi_t, _ = classical_evolution(Q, pi, t)
        l1 = float(np.abs(pi_t - pi).sum())
        print(f"  t={t}: ||pi(t) - pi_eq||_1 = {l1:.2e}")
        assert l1 < 1e-10, f"stationary distribution not preserved at t={t}"


def test_propagator_preserves_normalisation():
    """The unnormalised update must already sum to 1 (no renorm rescue)."""
    import scipy.linalg
    Q, pi = _setup()
    for t in (0.1, 0.5, 1.0):
        raw = scipy.linalg.expm(Q * t).T @ pi
        print(f"  t={t}: sum = {raw.sum():.10f}")
        assert abs(raw.sum() - 1.0) < 1e-10


def test_reweighting_is_exact_inverse_of_symmetrization():
    """sqrt(p * pi_eq) must recover pi(t) exactly from the symmetrized probs."""
    Q, pi = _setup()
    ok = pi > 1e-15
    for t in (0.1, 0.5, 1.0):
        pi_t, _ = classical_evolution(Q, pi, t)
        # probabilities the symmetrized circuit would measure
        psi = np.zeros_like(pi); psi[ok] = pi_t[ok] / np.sqrt(pi[ok])
        p = psi ** 2; p = p / p.sum()
        recovered = reweight_to_distribution(p, pi, len(pi))
        f = _hellinger(recovered, pi_t)
        print(f"  t={t}: F_H(recovered, pi(t)) = {f:.10f}")
        assert f > 1.0 - 1e-9, f"readout is not the inverse map (F_H={f})"


def test_wrong_reweighting_direction_is_rejected():
    """Guard: the old sqrt(p / pi_eq) form must NOT reproduce pi(t)."""
    Q, pi = _setup()
    ok = pi > 1e-15
    pi_t, _ = classical_evolution(Q, pi, 0.5)
    psi = np.zeros_like(pi); psi[ok] = pi_t[ok] / np.sqrt(pi[ok])
    p = psi ** 2; p = p / p.sum()
    wrong = np.zeros_like(pi)
    wrong[ok] = np.sqrt(p[ok] / pi[ok])
    wrong = wrong / wrong.sum()
    f = _hellinger(wrong, pi_t)
    print(f"  old sqrt(p/pi_eq) gives F_H = {f:.6f} (expect ~0.894, not 1.0)")
    assert f < 0.99, "the wrong readout should not score as correct"


def test_gapdh_register_encodes_empirical_probabilities():
    """Measured target probabilities must equal normalized codon counts."""
    reg = build_gapdh_register(n_qubits=6)
    expected = reg['weight_vector'] / reg['weight_vector'].sum()
    measured = np.asarray(reg['d_normalized']) ** 2
    assert np.isclose(measured.sum(), 1.0, atol=1e-14)
    assert np.allclose(measured, expected, atol=1e-14)
    assert np.allclose(reg['p_comp'], expected, atol=1e-14)


if __name__ == "__main__":
    for fn in (test_stationary_distribution_is_fixed,
               test_propagator_preserves_normalisation,
               test_reweighting_is_exact_inverse_of_symmetrization,
               test_wrong_reweighting_direction_is_rejected,
               test_gapdh_register_encodes_empirical_probabilities):
        print(f"\n{fn.__name__}")
        fn()
    print("\nAll readout/convention tests passed.")
