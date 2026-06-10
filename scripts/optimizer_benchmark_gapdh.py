"""
Optimizer Benchmark for AAE on GAPDH 4-Species Data
=====================================================
Tests 5 optimizers on the same AAE problem using the pooled GAPDH
codon frequencies (61 sense codons, 7 qubits, brickwall ansatz).

Saves convergence histories and final metrics to JSON for plotting.

Optimizers:
  1. L-BFGS-B   (quasi-Newton, gradient via finite differences)
  2. COBYLA      (gradient-free, trust-region)
  3. Nelder-Mead (gradient-free, simplex)
  4. SPSA        (stochastic perturbation, 2 evals/iter)
  5. Adam+ParamShift (1st-order gradient, parameter shift rule)

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/optimizer_benchmark_gapdh.py
"""

import os
import sys
import time
import json
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from data.gapdh_sequences import build_gapdh_register

# =========================================================================
# CONFIG
# =========================================================================
N_LAYERS = 8
SEED = 42
RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ITERS_LBFGS = 500
ITERS_COBYLA = 2000
ITERS_NM = 5000
ITERS_SPSA = 500
ITERS_ADAM = 200
EVAL_INTERVAL = 10  # record cost every N evaluations


# =========================================================================
# ANSATZ + COST
# =========================================================================

def build_brickwall_ansatz(n_qubits, n_layers, params):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)
    return qc


def get_statevector(params, n_qubits, n_layers):
    return np.array(Statevector.from_instruction(
        build_brickwall_ansatz(n_qubits, n_layers, params)).data)


def cost_fn(params, n_qubits, n_layers, target):
    sv = get_statevector(params, n_qubits, n_layers)
    return 1.0 - np.real(np.vdot(target, sv))


# =========================================================================
# OPTIMIZERS (each returns dict with history)
# =========================================================================

def run_lbfgs(n_q, n_l, target, p0):
    history = []
    eval_count = [0]
    def callback(params):
        c = cost_fn(params, n_q, n_l, target)
        eval_count[0] += 1
        history.append({'eval': eval_count[0], 'cost': float(c)})
    t0 = time.time()
    r = minimize(cost_fn, p0, args=(n_q, n_l, target), method='L-BFGS-B',
                 callback=callback,
                 options={'maxiter': ITERS_LBFGS, 'ftol': 1e-15, 'gtol': 1e-10})
    elapsed = time.time() - t0
    return {'name': 'L-BFGS', 'cost': float(r.fun), 'overlap': float(1 - r.fun),
            'evals': int(r.nfev), 'time': elapsed, 'history': history,
            'best_params': r.x.tolist()}


def run_cobyla(n_q, n_l, target, p0):
    history = []
    eval_count = [0]
    def callback(params):
        eval_count[0] += 1
        if eval_count[0] % EVAL_INTERVAL == 0:
            c = cost_fn(params, n_q, n_l, target)
            history.append({'eval': eval_count[0], 'cost': float(c)})
    t0 = time.time()
    r = minimize(cost_fn, p0, args=(n_q, n_l, target), method='COBYLA',
                 callback=callback,
                 options={'maxiter': ITERS_COBYLA, 'rhobeg': 0.5})
    elapsed = time.time() - t0
    # Record final cost
    history.append({'eval': int(r.nfev), 'cost': float(r.fun)})
    return {'name': 'COBYLA', 'cost': float(r.fun), 'overlap': float(1 - r.fun),
            'evals': int(r.nfev), 'time': elapsed, 'history': history,
            'best_params': r.x.tolist()}


def run_nelder_mead(n_q, n_l, target, p0):
    history = []
    eval_count = [0]
    best_cost = [float('inf')]
    def callback(params):
        eval_count[0] += 1
        if eval_count[0] % EVAL_INTERVAL == 0:
            c = cost_fn(params, n_q, n_l, target)
            if c < best_cost[0]:
                best_cost[0] = c
            history.append({'eval': eval_count[0], 'cost': float(c)})
    t0 = time.time()
    r = minimize(cost_fn, p0, args=(n_q, n_l, target), method='Nelder-Mead',
                 callback=callback,
                 options={'maxiter': ITERS_NM, 'xatol': 1e-10, 'fatol': 1e-12})
    elapsed = time.time() - t0
    history.append({'eval': int(r.nfev), 'cost': float(r.fun)})
    return {'name': 'Nelder-Mead', 'cost': float(r.fun), 'overlap': float(1 - r.fun),
            'evals': int(r.nfev), 'time': elapsed, 'history': history,
            'best_params': r.x.tolist()}


def run_spsa(n_q, n_l, target, p0):
    """Simultaneous Perturbation Stochastic Approximation."""
    history = []
    params = p0.copy()
    n_params = len(params)
    best_cost = float('inf')
    best_params = params.copy()

    a, c_val = 0.1, 0.1
    A_spsa = ITERS_SPSA * 0.1
    alpha_spsa, gamma_spsa = 0.602, 0.101

    t0 = time.time()
    total_evals = 0
    for k in range(ITERS_SPSA):
        a_k = a / (k + 1 + A_spsa) ** alpha_spsa
        c_k = c_val / (k + 1) ** gamma_spsa

        delta = 2 * np.random.randint(0, 2, size=n_params) - 1
        f_plus = cost_fn(params + c_k * delta, n_q, n_l, target)
        f_minus = cost_fn(params - c_k * delta, n_q, n_l, target)
        total_evals += 2

        g_hat = (f_plus - f_minus) / (2 * c_k * delta)
        params = params - a_k * g_hat

        current_cost = min(f_plus, f_minus)
        if current_cost < best_cost:
            best_cost = current_cost
            best_params = params.copy()

        if (k + 1) % EVAL_INTERVAL == 0 or k == 0:
            c = cost_fn(params, n_q, n_l, target)
            total_evals += 1
            history.append({'eval': total_evals, 'cost': float(c)})

    elapsed = time.time() - t0
    final_cost = cost_fn(best_params, n_q, n_l, target)
    history.append({'eval': total_evals + 1, 'cost': float(final_cost)})
    return {'name': 'SPSA', 'cost': float(final_cost), 'overlap': float(1 - final_cost),
            'evals': total_evals + 1, 'time': elapsed, 'history': history,
            'best_params': best_params.tolist()}


def run_adam_paramshift(n_q, n_l, target, p0):
    """Adam optimizer with parameter-shift gradients."""
    history = []
    params = p0.copy()
    n_params = len(params)
    best_cost = float('inf')
    best_params = params.copy()

    lr = 0.05
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(n_params)
    v = np.zeros(n_params)

    t0 = time.time()
    total_evals = 0
    for k in range(ITERS_ADAM):
        # Parameter-shift gradient
        grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += np.pi / 2
            params_minus[i] -= np.pi / 2
            f_plus = cost_fn(params_plus, n_q, n_l, target)
            f_minus = cost_fn(params_minus, n_q, n_l, target)
            grad[i] = (f_plus - f_minus) / 2.0
            total_evals += 2

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        current_cost = cost_fn(params, n_q, n_l, target)
        total_evals += 1
        if current_cost < best_cost:
            best_cost = current_cost
            best_params = params.copy()

        if (k + 1) % 5 == 0 or k == 0:
            history.append({'eval': total_evals, 'cost': float(current_cost)})

    elapsed = time.time() - t0
    final_cost = cost_fn(best_params, n_q, n_l, target)
    history.append({'eval': total_evals, 'cost': float(final_cost)})
    return {'name': 'Adam+ParamShift', 'cost': float(final_cost),
            'overlap': float(1 - final_cost),
            'evals': total_evals, 'time': elapsed, 'history': history,
            'best_params': best_params.tolist()}


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  OPTIMIZER BENCHMARK — GAPDH 4-Species (6 qubits, 8 layers)")
    print("=" * 70)

    # Build target state
    reg = build_gapdh_register(n_qubits=6)
    target = reg['d_normalized']
    n_q = reg['num_qubits']
    n_l = N_LAYERS
    n_params = n_q * n_l

    print(f"\n  Target: {reg['num_unique']} unique sense codons")
    print(f"  Qubits: {n_q},  Layers: {n_l},  Parameters: {n_params}")

    # Fixed random seed for reproducible initial parameters
    np.random.seed(SEED)
    p0 = np.random.uniform(0, 1, n_params)
    print(f"  Seed: {SEED}")

    optimizers = [
        ('L-BFGS',          run_lbfgs),
        ('COBYLA',          run_cobyla),
        ('Nelder-Mead',     run_nelder_mead),
        ('SPSA',            run_spsa),
        ('Adam+ParamShift', run_adam_paramshift),
    ]

    all_results = []
    for name, runner in optimizers:
        print(f"\n  Running {name}...")
        result = runner(n_q, n_l, target, p0.copy())
        all_results.append(result)
        print(f"    Cost: {result['cost']:.8f}  Overlap: {result['overlap']:.6f}  "
              f"Evals: {result['evals']}  Time: {result['time']:.1f}s")

    # Print summary table
    print(f"\n  {'Optimizer':<20} {'Cost':>12} {'Overlap':>10} {'Evals':>7} {'Time':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*7} {'-'*8}")
    for r in all_results:
        print(f"  {r['name']:<20} {r['cost']:>12.8f} {r['overlap']:>10.6f} "
              f"{r['evals']:>7d} {r['time']:>7.1f}s")

    # Save results (without large param arrays for cleaner JSON)
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != 'best_params'}
        save_results.append(save_r)

    out_path = os.path.join(RESULTS_DIR, 'optimizer_benchmark_gapdh.json')
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Also save the best params from L-BFGS for use in the main pipeline
    lbfgs_result = all_results[0]
    params_path = os.path.join(RESULTS_DIR, 'best_aae_params_gapdh.json')
    with open(params_path, 'w') as f:
        json.dump({
            'params': lbfgs_result['best_params'],
            'cost': lbfgs_result['cost'],
            'overlap': lbfgs_result['overlap'],
            'n_qubits': n_q,
            'n_layers': n_l,
        }, f, indent=2)
    print(f"  Best params saved to: {params_path}")


if __name__ == "__main__":
    main()
