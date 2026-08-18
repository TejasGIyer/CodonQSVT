"""Regression tests for the odd-channel sign in imaginary-time QSVT."""

import numpy as np

from src.qsvt_circuit_imagtime import combine_imagtime_amplitudes


def test_negative_spectrum_requires_negative_sinh_channel():
    """For lambda<0, singular-value sinh must be subtracted to give exp(t*lambda)."""
    eigenvalues = np.array([-2.0, -0.7, -0.1])
    t = 0.4
    cosh_channel = np.cosh(t * np.abs(eigenvalues))
    sinh_singular_channel = np.sinh(t * np.abs(eigenvalues))

    evolved = combine_imagtime_amplitudes(
        cosh_channel, sinh_singular_channel, 1.0, 1.0, sinh_sign=-1.0)

    assert np.allclose(evolved, np.exp(t * eigenvalues), atol=1e-14)
    assert np.all(evolved <= 1.0)


def test_positive_sinh_sign_would_evolve_backwards():
    eigenvalues = np.array([-1.5, -0.25])
    t = 0.3
    wrong = combine_imagtime_amplitudes(
        np.cosh(t * np.abs(eigenvalues)),
        np.sinh(t * np.abs(eigenvalues)),
        1.0, 1.0, sinh_sign=1.0)

    assert np.allclose(wrong, np.exp(-t * eigenvalues), atol=1e-14)
    assert np.all(wrong > 1.0)
