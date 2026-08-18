"""Scalar-shifted QSVT validation against the full GY94 CTMC.

This experiment is intentionally isolated from the production pipeline.  It
keeps a more accurate Pauli truncation and shifts it by ``-s I`` so its entire
spectrum is strictly negative.  The scalar shift changes only the norm of the
imaginary-time state; it cancels from the normalized codon distribution.

The default evaluator applies the exact Chebyshev polynomials used to generate
the QSP phase sequences to the shifted operator.  This is the ideal FTQC QSVT
algorithm, without the cost of expanding a very deep 13-qubit gate-level
statevector.  An optional gate checkpoint exercises that expansion explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cheb
from qiskit.quantum_info import SparsePauliOp
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.response import ComputeQSPResponse
from scipy.linalg import expm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gapdh_sequences import pooled_codon_frequencies
from src.constants import (
    GY94_KAPPA,
    GY94_OMEGA,
    GY94_V,
    N_DATA_QUBITS,
    PAULI_FULL_THRESHOLD,
)
from src.gy94_model import build_gy94_rate_matrix
from src.hamiltonian import (
    decompose_to_pauli,
    filter_pauli_op,
    reweight_to_distribution,
    symmetrize_to_hamiltonian,
)
from src.qsvt_angles_imagtime import (
    chebyshev_coefs_from_function,
    estimate_chebyshev_degree,
)
from src.qsvt_circuit_imagtime import assert_strictly_negative
from src.trotter import classical_evolution


DEFAULT_THRESHOLDS = (0.20, 0.10, 0.075, 0.05)
DEFAULT_SELECTED_THRESHOLD = 0.075
DEFAULT_TIMES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_EPSILON = 1e-3
DEFAULT_SHIFT_MARGIN = 1e-6
# pyqsp's Laurent phase finder normally perturbs the requested polynomial by
# about 1e-4 (``eps``/``suc`` capitalization).  That is harmless at small
# normalization but is amplified by 2*cosh(alpha*t) in long-time imaginary
# evolution.  Use near-machine-precision capitalization here and explicitly
# verify the synthesized phase response before accepting a result.
PHASE_CAPITALIZATION = 1e-14
PHASE_SYNTHESIS_TOLERANCE = 1e-12
IDENTITY_LABEL = "I" * N_DATA_QUBITS


@dataclass(frozen=True)
class ShiftedOperator:
    threshold: float
    unshifted_pauli: SparsePauliOp
    shifted_pauli: SparsePauliOp
    unshifted_matrix: np.ndarray
    shifted_matrix: np.ndarray
    shift: float
    lambda_max_before: float
    lambda_max_after: float
    n_terms: int
    alpha: float
    relative_spectral_error: float
    relative_frobenius_error: float


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    q = np.clip(np.asarray(q, dtype=float), 0.0, None)
    p /= p.sum()
    q /= q.sum()
    return float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    q = np.clip(np.asarray(q, dtype=float), 0.0, None)
    p /= p.sum()
    q /= q.sum()
    return float(0.5 * np.sum(np.abs(p - q)))


def skill_score(model_fidelity: float, baseline_fidelity: float) -> float | None:
    denom = 1.0 - baseline_fidelity
    if denom <= 1e-14:
        return None
    return float((model_fidelity - baseline_fidelity) / denom)


def make_delta_initial(pi_eq: np.ndarray, codons: list[str]) -> tuple[np.ndarray, str]:
    observed = np.flatnonzero(pi_eq > 0)
    index = int(observed[np.argmin(pi_eq[observed])])
    pi0 = np.zeros_like(pi_eq)
    pi0[index] = 1.0
    return pi0, f"delta on least-frequent observed codon {codons[index]}"


def symmetrized_state(pi0: np.ndarray, pi_eq: np.ndarray) -> np.ndarray:
    state = np.zeros(2**N_DATA_QUBITS, dtype=float)
    support = pi_eq > 0
    state[: len(pi0)][support] = pi0[support] / np.sqrt(pi_eq[support])
    norm = float(np.linalg.norm(state))
    if norm <= 0:
        raise ValueError("initial symmetrized state has zero norm")
    return state / norm


def measured_distribution(state: np.ndarray, pi_eq: np.ndarray) -> np.ndarray:
    probs = np.abs(np.asarray(state[: len(pi_eq)])) ** 2
    return reweight_to_distribution(probs, pi_eq, len(pi_eq))


def negative_physical_fraction(state: np.ndarray, pi_eq: np.ndarray) -> tuple[int, float]:
    """Check whether inverse symmetrization would contain negative mass.

    Measurement only reveals magnitudes.  A large negative signed amplitude
    could otherwise be hidden by the square-root readout map.
    """
    physical = np.sqrt(pi_eq) * np.real(np.asarray(state[: len(pi_eq)]))
    n_negative = int(np.sum(physical < -1e-12))
    negative_l1 = float(-np.sum(physical[physical < 0]))
    denominator = float(np.sum(np.abs(physical)))
    fraction = negative_l1 / denominator if denominator > 0 else 0.0
    return n_negative, float(max(0.0, fraction))


def add_scalar_shift(
    pauli_op: SparsePauliOp,
    full_hamiltonian: np.ndarray,
    threshold: float,
    margin: float,
) -> ShiftedOperator:
    unshifted = np.real_if_close(pauli_op.to_matrix()).astype(float)
    eigenvalues = np.linalg.eigvalsh(unshifted)
    lambda_max_before = float(eigenvalues[-1])
    shift = max(0.0, lambda_max_before + margin)

    labels = list(pauli_op.paulis.to_labels())
    coefficients = np.asarray(pauli_op.coeffs, dtype=complex).copy()
    if IDENTITY_LABEL in labels:
        coefficients[labels.index(IDENTITY_LABEL)] -= shift
    else:
        labels.append(IDENTITY_LABEL)
        coefficients = np.append(coefficients, -shift)

    shifted_pauli = SparsePauliOp.from_list(list(zip(labels, coefficients))).simplify(
        atol=1e-14
    )
    shifted = np.real_if_close(shifted_pauli.to_matrix()).astype(float)
    lambda_max_after = float(np.linalg.eigvalsh(shifted)[-1])
    alpha = float(np.sum(np.abs(shifted_pauli.coeffs)))

    spectral_denominator = float(np.linalg.norm(full_hamiltonian, 2))
    frobenius_denominator = float(np.linalg.norm(full_hamiltonian, "fro"))
    relative_spectral_error = float(
        np.linalg.norm(unshifted - full_hamiltonian, 2) / spectral_denominator
    )
    relative_frobenius_error = float(
        np.linalg.norm(unshifted - full_hamiltonian, "fro") / frobenius_denominator
    )

    ok, checked_lambda_max = assert_strictly_negative(shifted_pauli)
    if not ok or abs(checked_lambda_max - lambda_max_after) > 1e-9:
        raise AssertionError("shifted Pauli operator failed strict-negativity check")

    return ShiftedOperator(
        threshold=float(threshold),
        unshifted_pauli=pauli_op,
        shifted_pauli=shifted_pauli,
        unshifted_matrix=unshifted,
        shifted_matrix=shifted,
        shift=float(shift),
        lambda_max_before=lambda_max_before,
        lambda_max_after=lambda_max_after,
        n_terms=len(shifted_pauli),
        alpha=alpha,
        relative_spectral_error=relative_spectral_error,
        relative_frobenius_error=relative_frobenius_error,
    )


def _qsvt_polynomial_data(alpha: float, t: float, epsilon: float) -> dict:
    """Return high-precision phases and their requested polynomials.

    The main pipeline's default Laurent capitalization is intentionally not
    reused: a 1e-4 phase-response perturbation would be magnified at large
    ``2*cosh(alpha*t)``.  These phases are synthesized with a 1e-14
    capitalization and are independently reconstructed with pyqsp below.
    """
    tau = alpha * t
    norm_factor = 2.0 * np.cosh(tau)
    f_cosh = lambda x: np.cosh(tau * x) / norm_factor
    f_sinh = lambda x: np.sinh(tau * x) / norm_factor
    degree_cosh = estimate_chebyshev_degree(tau, epsilon, "even")
    degree_sinh = estimate_chebyshev_degree(tau, epsilon, "odd")
    coefficients_cosh = chebyshev_coefs_from_function(
        f_cosh, degree_cosh, parity="even"
    )
    coefficients_sinh = chebyshev_coefs_from_function(
        f_sinh, degree_sinh, parity="odd"
    )

    grid = np.linspace(-1.0, 1.0, 2001)
    max_cosh = float(np.max(np.abs(cheb.chebval(grid, coefficients_cosh))))
    max_sinh = float(np.max(np.abs(cheb.chebval(grid, coefficients_sinh))))
    safety = 1.0 / 1.02
    if max_cosh > 1.0:
        coefficients_cosh *= safety / max_cosh
    if max_sinh > 1.0:
        coefficients_sinh *= safety / max_sinh

    phases_cosh = np.asarray(
        QuantumSignalProcessingPhases(
            coefficients_cosh.tolist(),
            signal_operator="Wx",
            method="laurent",
            eps=PHASE_CAPITALIZATION,
            suc=1.0 - PHASE_CAPITALIZATION,
            tolerance=PHASE_SYNTHESIS_TOLERANCE,
        ),
        dtype=float,
    )
    phases_sinh = np.asarray(
        QuantumSignalProcessingPhases(
            coefficients_sinh.tolist(),
            signal_operator="Wx",
            method="laurent",
            eps=PHASE_CAPITALIZATION,
            suc=1.0 - PHASE_CAPITALIZATION,
            tolerance=PHASE_SYNTHESIS_TOLERANCE,
        ),
        dtype=float,
    )

    negative_grid = np.linspace(-1.0, 0.0, 2001)
    target_grid = (
        cheb.chebval(negative_grid, coefficients_cosh) * norm_factor
        + cheb.chebval(negative_grid, coefficients_sinh) * norm_factor
    )
    target_error = float(
        np.max(np.abs(target_grid - np.exp(tau * negative_grid)))
    )
    info = {
        "tau": float(tau),
        "norm_factor_cosh": float(norm_factor),
        "norm_factor_sinh": float(norm_factor),
        "cosh_degree": int(degree_cosh),
        "sinh_degree": int(degree_sinh),
        "approx_error_on_neg_interval": target_error,
    }

    return {
        "phases_cosh": phases_cosh,
        "phases_sinh": phases_sinh,
        "coefficients_cosh": coefficients_cosh,
        "coefficients_sinh": coefficients_sinh,
        "info": info,
    }


def ideal_qsvt_action(
    shifted_matrix: np.ndarray,
    alpha: float,
    state: np.ndarray,
    t: float,
    epsilon: float,
) -> tuple[np.ndarray, dict]:
    """Apply the synthesized ideal-QSVT polynomial to ``state``.

    For a strictly negative Hermitian operator the existing circuit's global
    odd-channel correction restores the signed odd polynomial.  Evaluating the
    even and signed odd Chebyshev series on the eigenvalues is therefore the
    ideal FTQC action of the generated phase sequences.
    """
    if t == 0:
        return np.asarray(state, dtype=float).copy(), {
            "tau": 0.0,
            "cosh_degree": 0,
            "sinh_degree": 0,
            "n_cosh_phases": 0,
            "n_sinh_phases": 0,
            "target_polynomial_error_actual_spectrum": 0.0,
            "phase_response_error_actual_spectrum": 0.0,
            "max_channel_phase_response_error_unscaled": 0.0,
            "max_channel_phase_response_imaginary": 0.0,
            "approx_error_on_neg_interval": 0.0,
        }

    data = _qsvt_polynomial_data(alpha, t, epsilon)
    info = data["info"]
    eigenvalues, eigenvectors = np.linalg.eigh(shifted_matrix)
    normalized_eigenvalues = eigenvalues / alpha
    if np.min(normalized_eigenvalues) < -1.0 - 1e-10:
        raise AssertionError("block-encoded spectrum extends below -1")
    if np.max(normalized_eigenvalues) >= 0.0:
        raise AssertionError("shifted spectrum is not strictly negative")

    target_cosh = cheb.chebval(
        normalized_eigenvalues, data["coefficients_cosh"]
    )
    target_sinh = cheb.chebval(
        normalized_eigenvalues, data["coefficients_sinh"]
    )
    response_cosh_complex = ComputeQSPResponse(
        normalized_eigenvalues,
        data["phases_cosh"],
        signal_operator="Wx",
    )["pdat"]
    response_sinh_complex = ComputeQSPResponse(
        normalized_eigenvalues,
        data["phases_sinh"],
        signal_operator="Wx",
    )["pdat"]
    response_cosh = np.real(response_cosh_complex)
    response_sinh = np.real(response_sinh_complex)

    # For the negative spectrum, this signed response is what the circuit's
    # global odd-channel correction restores from the singular-value response.
    phase_response_values = (
        response_cosh * info["norm_factor_cosh"]
        + response_sinh * info["norm_factor_sinh"]
    )
    target_polynomial_values = (
        target_cosh * info["norm_factor_cosh"]
        + target_sinh * info["norm_factor_sinh"]
    )
    exact_values = np.exp(t * eigenvalues)
    target_spectrum_error = float(
        np.max(np.abs(target_polynomial_values - exact_values))
    )
    phase_spectrum_error = float(
        np.max(np.abs(phase_response_values - exact_values))
    )
    max_channel_response_error = float(
        max(
            np.max(np.abs(response_cosh - target_cosh)),
            np.max(np.abs(response_sinh - target_sinh)),
        )
    )
    max_channel_imaginary = float(
        max(
            np.max(np.abs(np.imag(response_cosh_complex))),
            np.max(np.abs(np.imag(response_sinh_complex))),
        )
    )
    transformed = eigenvectors.T.conj() @ state
    output = eigenvectors @ (phase_response_values * transformed)

    diagnostics = {
        "tau": float(info["tau"]),
        "cosh_degree": int(info["cosh_degree"]),
        "sinh_degree": int(info["sinh_degree"]),
        "n_cosh_phases": int(len(data["phases_cosh"])),
        "n_sinh_phases": int(len(data["phases_sinh"])),
        "target_polynomial_error_actual_spectrum": target_spectrum_error,
        "phase_response_error_actual_spectrum": phase_spectrum_error,
        "max_channel_phase_response_error_unscaled": max_channel_response_error,
        "max_channel_phase_response_imaginary": max_channel_imaginary,
        "approx_error_on_neg_interval": float(info["approx_error_on_neg_interval"]),
    }
    return np.real_if_close(output), diagnostics


def exact_evolved_distribution(
    matrix: np.ndarray, state: np.ndarray, pi_eq: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray]:
    evolved = expm(matrix * t) @ state
    return measured_distribution(evolved, pi_eq), evolved


def build_model() -> dict:
    frequencies = pooled_codon_frequencies()
    Q, codons, pi_eq, q_info = build_gy94_rate_matrix(
        frequencies, kappa=GY94_KAPPA, V=GY94_V
    )
    pi_eq = pi_eq / pi_eq.sum()
    H, h_info = symmetrize_to_hamiltonian(Q, pi_eq, n_qubits=N_DATA_QUBITS)
    pauli_full, pauli_info = decompose_to_pauli(
        H, n_qubits=N_DATA_QUBITS, threshold=PAULI_FULL_THRESHOLD
    )
    pi0, init_label = make_delta_initial(pi_eq, codons)
    psi0 = symmetrized_state(pi0, pi_eq)
    return {
        "Q": Q,
        "codons": codons,
        "pi_eq": pi_eq,
        "H": H,
        "q_info": q_info,
        "h_info": h_info,
        "pauli_full": pauli_full,
        "pauli_info": pauli_info,
        "pi0": pi0,
        "psi0": psi0,
        "init_label": init_label,
    }


def scan_thresholds(
    model: dict,
    thresholds: Iterable[float],
    times: Iterable[float],
    shift_margin: float,
) -> tuple[dict[float, ShiftedOperator], list[dict]]:
    operators: dict[float, ShiftedOperator] = {}
    rows: list[dict] = []
    for threshold in thresholds:
        pauli_op, _ = filter_pauli_op(model["pauli_full"], threshold)
        shifted = add_scalar_shift(pauli_op, model["H"], threshold, shift_margin)
        operators[float(threshold)] = shifted
        for t in times:
            pi_classical, _ = classical_evolution(model["Q"], model["pi0"], t)
            pi_truncated, state_truncated = exact_evolved_distribution(
                shifted.unshifted_matrix, model["psi0"], model["pi_eq"], t
            )
            n_negative, negative_fraction = negative_physical_fraction(
                state_truncated, model["pi_eq"]
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "t": float(t),
                    "n_terms": int(shifted.n_terms),
                    "shift": float(shifted.shift),
                    "alpha": float(shifted.alpha),
                    "lambda_max_before": float(shifted.lambda_max_before),
                    "lambda_max_after": float(shifted.lambda_max_after),
                    "relative_spectral_error": shifted.relative_spectral_error,
                    "relative_frobenius_error": shifted.relative_frobenius_error,
                    "f_exact_truncated_vs_ctmc": hellinger_fidelity(
                        pi_truncated, pi_classical
                    ),
                    "tv_exact_truncated_vs_ctmc": total_variation(
                        pi_truncated, pi_classical
                    ),
                    "negative_physical_entries": n_negative,
                    "negative_physical_l1_fraction": negative_fraction,
                }
            )
    return operators, rows


def run_selected_qsvt(
    model: dict,
    shifted: ShiftedOperator,
    times: Iterable[float],
    epsilon: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    probabilities: list[dict] = []
    for t in times:
        started = time.time()
        pi_classical, _ = classical_evolution(model["Q"], model["pi0"], t)
        pi_full_h, _ = exact_evolved_distribution(
            model["H"], model["psi0"], model["pi_eq"], t
        )
        pi_unshifted, state_unshifted = exact_evolved_distribution(
            shifted.unshifted_matrix, model["psi0"], model["pi_eq"], t
        )
        pi_shifted, state_shifted = exact_evolved_distribution(
            shifted.shifted_matrix, model["psi0"], model["pi_eq"], t
        )
        qsvt_state, qsvt_info = ideal_qsvt_action(
            shifted.shifted_matrix,
            shifted.alpha,
            model["psi0"],
            t,
            epsilon,
        )
        pi_qsvt = measured_distribution(qsvt_state, model["pi_eq"])
        identity_fidelity = hellinger_fidelity(model["pi0"], pi_classical)
        qsvt_fidelity = hellinger_fidelity(pi_qsvt, pi_classical)
        n_negative_exact, negative_fraction_exact = negative_physical_fraction(
            state_unshifted, model["pi_eq"]
        )
        n_negative_qsvt, negative_fraction_qsvt = negative_physical_fraction(
            qsvt_state, model["pi_eq"]
        )

        rows.append(
            {
                "t": float(t),
                "f_qsvt_vs_ctmc": qsvt_fidelity,
                "tv_qsvt_vs_ctmc": total_variation(pi_qsvt, pi_classical),
                "f_exact_truncated_vs_ctmc": hellinger_fidelity(
                    pi_unshifted, pi_classical
                ),
                "tv_exact_truncated_vs_ctmc": total_variation(
                    pi_unshifted, pi_classical
                ),
                "f_qsvt_vs_exact_shifted": hellinger_fidelity(pi_qsvt, pi_shifted),
                "tv_qsvt_vs_exact_shifted": total_variation(pi_qsvt, pi_shifted),
                "f_shifted_vs_unshifted": hellinger_fidelity(
                    pi_shifted, pi_unshifted
                ),
                "tv_shifted_vs_unshifted": total_variation(
                    pi_shifted, pi_unshifted
                ),
                "f_full_h_vs_ctmc": hellinger_fidelity(pi_full_h, pi_classical),
                "tv_full_h_vs_ctmc": total_variation(pi_full_h, pi_classical),
                "f_identity_vs_ctmc": identity_fidelity,
                "skill_qsvt_vs_identity": skill_score(qsvt_fidelity, identity_fidelity),
                "qsvt_shifted_norm2": float(np.vdot(qsvt_state, qsvt_state).real),
                "exact_shifted_norm2": float(np.vdot(state_shifted, state_shifted).real),
                "exact_unshifted_norm2": float(
                    np.vdot(state_unshifted, state_unshifted).real
                ),
                "negative_physical_entries_exact_truncated": n_negative_exact,
                "negative_physical_l1_fraction_exact_truncated": negative_fraction_exact,
                "negative_physical_entries_qsvt": n_negative_qsvt,
                "negative_physical_l1_fraction_qsvt": negative_fraction_qsvt,
                "tau": qsvt_info["tau"],
                "cosh_degree": qsvt_info["cosh_degree"],
                "sinh_degree": qsvt_info["sinh_degree"],
                "n_cosh_phases": qsvt_info["n_cosh_phases"],
                "n_sinh_phases": qsvt_info["n_sinh_phases"],
                "polynomial_error_actual_spectrum": qsvt_info[
                    "target_polynomial_error_actual_spectrum"
                ],
                "phase_response_error_actual_spectrum": qsvt_info[
                    "phase_response_error_actual_spectrum"
                ],
                "max_channel_phase_response_error_unscaled": qsvt_info[
                    "max_channel_phase_response_error_unscaled"
                ],
                "max_channel_phase_response_imaginary": qsvt_info[
                    "max_channel_phase_response_imaginary"
                ],
                "polynomial_error_negative_interval": qsvt_info[
                    "approx_error_on_neg_interval"
                ],
                "eval_time_s": float(time.time() - started),
            }
        )
        probabilities.append(
            {
                "t": float(t),
                "classical_ctmc": pi_classical.tolist(),
                "ideal_qsvt": pi_qsvt.tolist(),
                "exact_truncated": pi_unshifted.tolist(),
                "identity": model["pi0"].tolist(),
            }
        )
        print(
            f"  t={t:>4.2f}  F(QSVT,CTMC)={qsvt_fidelity:.6f}  "
            f"F(identity,CTMC)={identity_fidelity:.6f}  "
            f"F(QSVT,exact H_tau)={rows[-1]['f_qsvt_vs_exact_shifted']:.10f}"
        )
    return rows, probabilities


def run_gate_checkpoint(
    model: dict, shifted: ShiftedOperator, t: float, epsilon: float
) -> dict:
    """Optional, very expensive gate-expanded statevector checkpoint."""
    from src.block_encoding import build_simple_block_encoding
    from scripts.far_from_equilibrium import evaluate_qsvt_at_t, prepare_state_circuit

    print(f"\nBuilding optional gate-level checkpoint at t={t} ...")
    started = time.time()
    block_encoding, alpha, be_info = build_simple_block_encoding(
        shifted.shifted_pauli, n_data_qubits=N_DATA_QUBITS
    )
    if abs(alpha - shifted.alpha) > 1e-10:
        raise AssertionError("block-encoding alpha differs from shifted-operator alpha")
    initial_circuit = prepare_state_circuit(
        model["pi0"], model["pi_eq"], n_qubits=N_DATA_QUBITS
    )
    f_b, f_h, tv, norm2, probs_rw, n_phases = evaluate_qsvt_at_t(
        block_encoding,
        initial_circuit,
        model["Q"],
        model["pi0"],
        model["pi_eq"],
        alpha,
        be_info["n_ancilla"],
        t,
    )
    ideal_state, _ = ideal_qsvt_action(
        shifted.shifted_matrix, shifted.alpha, model["psi0"], t, epsilon
    )
    ideal_probs = measured_distribution(ideal_state, model["pi_eq"])
    return {
        "t": float(t),
        "f_gate_qsvt_vs_ctmc": float(f_h),
        "f_bhattacharyya_gate_qsvt_vs_ctmc": float(f_b),
        "tv_gate_qsvt_vs_ctmc": float(tv),
        "f_gate_vs_ideal_polynomial": hellinger_fidelity(probs_rw, ideal_probs),
        "tv_gate_vs_ideal_polynomial": total_variation(probs_rw, ideal_probs),
        "gate_qsvt_norm2": float(norm2),
        "n_phases": int(n_phases),
        "be_ancilla": int(be_info["n_ancilla"]),
        "be_depth": int(be_info["depth"]),
        "elapsed_s": float(time.time() - started),
    }


def acceptance_checks(rows: list[dict], shifted: ShiftedOperator) -> dict:
    nonzero_rows = [row for row in rows if row["t"] > 0]
    checks = {
        "strictly_negative_shifted_spectrum": shifted.lambda_max_after < 0,
        "full_h_matches_ctmc": min(row["f_full_h_vs_ctmc"] for row in rows)
        > 1.0 - 1e-10,
        "scalar_shift_preserves_distribution": min(
            row["f_shifted_vs_unshifted"] for row in rows
        )
        > 1.0 - 1e-10,
        "ideal_qsvt_matches_exact_shifted_operator": min(
            row["f_qsvt_vs_exact_shifted"] for row in rows
        )
        > 1.0 - 1e-8,
        "synthesized_phase_response_error_below_1e_4": max(
            row["phase_response_error_actual_spectrum"] for row in rows
        )
        < 1e-4,
        "qsvt_tracks_full_ctmc_above_0_94": min(
            row["f_qsvt_vs_ctmc"] for row in rows
        )
        > 0.94,
        "qsvt_beats_identity_at_all_nonzero_times": all(
            row["skill_qsvt_vs_identity"] is not None
            and row["skill_qsvt_vs_identity"] > 0
            for row in nonzero_rows
        ),
        "no_hidden_negative_physical_mass": max(
            row["negative_physical_l1_fraction_exact_truncated"] for row in rows
        )
        < 1e-10,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    output_dir: Path,
    selected_rows: list[dict],
    scan_rows: list[dict],
    thresholds: Iterable[float],
) -> None:
    times = np.array([row["t"] for row in selected_rows])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(times, [r["f_qsvt_vs_ctmc"] for r in selected_rows], "o-", label="Shifted ideal QSVT")
    ax.plot(times, [r["f_exact_truncated_vs_ctmc"] for r in selected_rows], "--", label=r"Exact $e^{H_\tau t}$")
    ax.plot(times, [r["f_identity_vs_ctmc"] for r in selected_rows], ":", label="Do nothing")
    ax.set_ylabel("Hellinger fidelity vs CTMC")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(times, [r["tv_qsvt_vs_ctmc"] for r in selected_rows], "o-", label="Shifted ideal QSVT")
    ax.plot(times, [r["tv_exact_truncated_vs_ctmc"] for r in selected_rows], "--", label=r"Exact $e^{H_\tau t}$")
    ax.set_ylabel("Total variation distance")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    skill_times = [r["t"] for r in selected_rows if r["skill_qsvt_vs_identity"] is not None]
    skills = [r["skill_qsvt_vs_identity"] for r in selected_rows if r["skill_qsvt_vs_identity"] is not None]
    ax.plot(skill_times, skills, "o-")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Skill over do-nothing baseline")
    ax.set_xlabel("Evolution time")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.semilogy(times, [max(1e-16, 1-r["f_qsvt_vs_exact_shifted"]) for r in selected_rows], "o-", label="1 - F(QSVT, exact shifted)")
    ax.semilogy(times, [max(1e-16, r["phase_response_error_actual_spectrum"]) for r in selected_rows], "s--", label="Synthesized phase-response error")
    ax.set_ylabel("Internal algorithm error")
    ax.set_xlabel("Evolution time")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle("Scalar-shifted QSVT validation (38-term operator)", fontsize=14)
    fig.savefig(output_dir / "shifted_qsvt_trajectory.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for threshold in thresholds:
        subset = [row for row in scan_rows if abs(row["threshold"] - threshold) < 1e-12]
        ax.plot(
            [row["t"] for row in subset],
            [row["f_exact_truncated_vs_ctmc"] for row in subset],
            "o-",
            label=f"cutoff={threshold:g} ({subset[0]['n_terms']} terms)",
        )
    ax.set_xlabel("Evolution time")
    ax.set_ylabel(r"Fidelity of exact $e^{H_\tau t}$ vs full CTMC")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Accuracy gained by retaining more Pauli terms")
    fig.savefig(output_dir / "threshold_accuracy_scan.png", dpi=220)
    plt.close(fig)


def write_summary(
    path: Path,
    selected: ShiftedOperator,
    rows: list[dict],
    checks: dict,
    gate_checkpoint: dict | None,
) -> None:
    nonzero = [row for row in rows if row["t"] > 0]
    lines = [
        "SCALAR-SHIFTED QSVT VALIDATION SUMMARY",
        "=" * 48,
        f"Selected coefficient cutoff: {selected.threshold}",
        f"Retained Pauli terms: {selected.n_terms}",
        f"Scalar shift s: {selected.shift:.9f}",
        f"lambda_max before shift: {selected.lambda_max_before:+.9f}",
        f"lambda_max after shift: {selected.lambda_max_after:+.9e}",
        f"Block-encoding alpha: {selected.alpha:.9f}",
        f"Relative spectral truncation error: {selected.relative_spectral_error:.6f}",
        "",
        f"Minimum QSVT-vs-CTMC fidelity: {min(r['f_qsvt_vs_ctmc'] for r in rows):.6f}",
        f"Minimum QSVT-vs-exact-shifted fidelity: {min(r['f_qsvt_vs_exact_shifted'] for r in rows):.12f}",
        f"Minimum skill over identity (t>0): {min(r['skill_qsvt_vs_identity'] for r in nonzero):.6f}",
        f"Maximum QSVT-vs-CTMC TV distance: {max(r['tv_qsvt_vs_ctmc'] for r in rows):.6f}",
        f"Maximum hidden negative physical mass: {max(r['negative_physical_l1_fraction_exact_truncated'] for r in rows):.3e}",
        "",
        "Acceptance checks:",
    ]
    lines.extend(f"  {'PASS' if value else 'FAIL'}  {name}" for name, value in checks.items())
    if gate_checkpoint is not None:
        lines.extend(
            [
                "",
                "Optional gate-level checkpoint:",
                json.dumps(gate_checkpoint, indent=2),
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The scalar shift makes the truncated operator strictly negative while",
            "leaving every normalized evolved codon distribution unchanged.  The",
            "synthesized high-precision QSP phase response reproduces the exact shifted",
            "exponential,",
            "and the 38-term trajectory tracks the full GY94 CTMC across the tested",
            "time range.  This is FTQC algorithm proof-of-concept evidence; it is not",
            "a near-term hardware or quantum-speedup claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="comma-separated coefficient cutoffs for the exact scan",
    )
    parser.add_argument(
        "--selected-threshold",
        type=float,
        default=DEFAULT_SELECTED_THRESHOLD,
        help="cutoff used for the ideal-QSVT trajectory",
    )
    parser.add_argument(
        "--times",
        default=",".join(str(value) for value in DEFAULT_TIMES),
        help="comma-separated evolution times",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--shift-margin", type=float, default=DEFAULT_SHIFT_MARGIN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--gate-checkpoint",
        type=float,
        default=None,
        help="optional expensive gate-expanded statevector checkpoint time",
    )
    args = parser.parse_args()

    thresholds = parse_float_list(args.thresholds)
    times = parse_float_list(args.times)
    if args.selected_threshold not in thresholds:
        thresholds = tuple(thresholds) + (args.selected_threshold,)
    if any(t < 0 for t in times):
        raise ValueError("evolution times must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building the GY94 CTMC and full symmetrized Hamiltonian ...")
    model = build_model()
    print(f"Initial state: {model['init_label']}")
    print("\nScanning Pauli cutoffs and scalar shifts ...")
    operators, scan_rows = scan_thresholds(
        model, thresholds, times, args.shift_margin
    )
    for threshold in thresholds:
        op = operators[float(threshold)]
        print(
            f"  cutoff={threshold:g}: terms={op.n_terms:>2}, "
            f"lambda_max={op.lambda_max_before:+.6f}, shift={op.shift:.6f}, "
            f"alpha={op.alpha:.6f}, rel_spec_err={op.relative_spectral_error:.4f}"
        )

    selected = operators[float(args.selected_threshold)]
    print(
        f"\nRunning ideal-QSVT trajectory for cutoff {selected.threshold:g} "
        f"({selected.n_terms} terms) ..."
    )
    selected_rows, probabilities = run_selected_qsvt(
        model, selected, times, args.epsilon
    )
    checks = acceptance_checks(selected_rows, selected)

    gate_result = None
    if args.gate_checkpoint is not None:
        gate_result = run_gate_checkpoint(
            model, selected, args.gate_checkpoint, args.epsilon
        )

    payload = {
        "config": {
            "experiment": "scalar_shifted_qsvt_validation",
            "interpretation": "ideal FTQC QSVT polynomial with generated QSP phases",
            "kappa": GY94_KAPPA,
            "omega": GY94_OMEGA,
            "V": GY94_V,
            "n_data_qubits": N_DATA_QUBITS,
            "initial_state": model["init_label"],
            "thresholds": list(thresholds),
            "selected_threshold": selected.threshold,
            "times": list(times),
            "epsilon": args.epsilon,
            "shift_margin": args.shift_margin,
            "phase_capitalization": PHASE_CAPITALIZATION,
            "phase_synthesis_tolerance": PHASE_SYNTHESIS_TOLERANCE,
        },
        "selected_operator": {
            "threshold": selected.threshold,
            "n_terms": selected.n_terms,
            "shift": selected.shift,
            "lambda_max_before": selected.lambda_max_before,
            "lambda_max_after": selected.lambda_max_after,
            "alpha": selected.alpha,
            "be_ancilla_qubits": max(1, math.ceil(math.log2(selected.n_terms))),
            "relative_spectral_error": selected.relative_spectral_error,
            "relative_frobenius_error": selected.relative_frobenius_error,
        },
        "acceptance_checks": checks,
        "trajectory_rows": selected_rows,
        "trajectory_probabilities": probabilities,
        "threshold_scan_rows": scan_rows,
        "optional_gate_checkpoint": gate_result,
    }

    json_path = args.output_dir / "shifted_qsvt_validation.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.output_dir / "shifted_qsvt_trajectory.csv", selected_rows)
    write_csv(args.output_dir / "threshold_accuracy_scan.csv", scan_rows)
    make_plots(args.output_dir, selected_rows, scan_rows, thresholds)
    write_summary(
        args.output_dir / "SUMMARY.txt", selected, selected_rows, checks, gate_result
    )

    print("\nAcceptance checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nArtifacts written to: {args.output_dir}")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
