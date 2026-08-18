"""Focused regression tests for the isolated shifted-QSVT experiment."""

import numpy as np
from scipy.linalg import expm

from shifted_qsvt_validation.run_validation import (
    hellinger_fidelity,
    ideal_qsvt_action,
)


def _normalized_probabilities(state):
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def test_scalar_identity_shift_preserves_normalized_distribution():
    matrix = np.array([[-0.7, 0.2], [0.2, -0.3]])
    state = np.array([1.0, 0.0])
    shift = 0.6
    for t in (0.1, 0.5, 1.0, 2.0):
        unshifted = expm(matrix * t) @ state
        shifted = expm((matrix - shift * np.eye(2)) * t) @ state
        assert hellinger_fidelity(
            _normalized_probabilities(unshifted),
            _normalized_probabilities(shifted),
        ) > 1.0 - 1e-12


def test_ideal_qsvt_polynomial_matches_exact_negative_exponential():
    matrix = np.diag([-0.9, -0.2])
    alpha = 1.1
    state = np.array([np.sqrt(0.3), np.sqrt(0.7)])
    for t in (0.1, 0.5, 1.0, 2.0):
        qsvt, _ = ideal_qsvt_action(matrix, alpha, state, t, epsilon=1e-3)
        exact = expm(matrix * t) @ state
        assert hellinger_fidelity(
            _normalized_probabilities(qsvt),
            _normalized_probabilities(exact),
        ) > 1.0 - 1e-10
