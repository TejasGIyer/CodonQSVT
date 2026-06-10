"""Step 2 (AAE): Approximate Amplitude Encoding — Brickwall ansatz + L-BFGS training."""

import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from src.constants import AAE_RANDOM_SEED 

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


def statevector_from_params(params, n_qubits, n_layers):
    return np.array(Statevector.from_instruction(build_brickwall_ansatz(n_qubits, n_layers, params)).data)


def cost_function(params, n_qubits, n_layers, target_state):
    sv = statevector_from_params(params, n_qubits, n_layers)
    return 1.0 - np.real(np.vdot(target_state, sv))


def train_pqc(n_qubits, n_layers, target_state, n_trials=6, maxiter=5000):
    n_params = n_qubits * n_layers
    best_params, best_cost = None, float('inf')

    rng = np.random.default_rng(AAE_RANDOM_SEED)
    for trial in range(n_trials):
        #params_init = np.random.uniform(0, 1, n_params)
        params_init = rng.uniform(0, 1, n_params)
        result = minimize(cost_function, params_init, args=(n_qubits, n_layers, target_state),
                          method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
        print(f"  Trial {trial+1}/{n_trials}: cost={result.fun:.8f}, iters={result.nit}, overlap={1-result.fun:.6f}")
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x.copy()

    return best_params, best_cost


def aae_encode(step1_result, n_layers=4, n_trials=6, maxiter=5000):
    n_q = step1_result['num_qubits']
    d = step1_result['d_normalized']

    print(f"\n  Config: {n_q} qubits, {n_layers} layers, {n_q * n_layers} params, {n_trials} trials")
    best_params, best_cost = train_pqc(n_q, n_layers, d, n_trials, maxiter)

    trained_circuit = build_brickwall_ansatz(n_q, n_layers, best_params)
    trained_circuit_meas = QuantumCircuit(n_q, n_q)
    trained_circuit_meas.compose(trained_circuit, inplace=True)
    trained_circuit_meas.measure(range(n_q), range(n_q))

    trained_sv = Statevector.from_instruction(trained_circuit)
    gc = dict(trained_circuit.count_ops())

    return {
        'encoding_type': 'aae', 'circuit': trained_circuit, 'circuit_meas': trained_circuit_meas,
        'initial_sv': trained_sv, 'initial_dm': DensityMatrix(trained_sv),
        'target_sv': Statevector(d), 'target_dm': DensityMatrix(Statevector(d)),
        'num_qubits': n_q, 'best_params': best_params, 'best_cost': best_cost,
        'overlap': abs(np.vdot(d, trained_sv.data)), 'n_layers': n_layers,
        'logical_cnot': gc.get('cx', 0), 'logical_ry': gc.get('ry', 0),
        'logical_total': gc.get('cx', 0) + gc.get('ry', 0),
    }


def print_step2(step1_result, step2_result):
    n_q = step2_result['num_qubits']
    d = step1_result['d_normalized']
    sv = step2_result['initial_sv']

    print("\n" + "=" * 65)
    print("STEP 2: APPROXIMATE AMPLITUDE ENCODING (AAE)")
    print("=" * 65)
    print(f"\n  Qubits: {n_q}   Layers: {step2_result['n_layers']}   Params: {n_q * step2_result['n_layers']}")
    print(f"  Gates: {step2_result['logical_ry']} Ry + {step2_result['logical_cnot']} CNOT = {step2_result['logical_total']}")
    print(f"  Depth: {step2_result['circuit'].depth()}")
    print(f"  Cost: {step2_result['best_cost']:.8f}   Overlap: {step2_result['overlap']:.6f}")

    probs_t = d ** 2
    probs_a = np.abs(sv.data) ** 2
    max_delta = 0

    print(f"\n  {'Basis':>9}  {'p_target':>9}  {'p_actual':>9}  {'|Δp|':>8}  Codon")
    print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*6}")
    for e in step1_result['unique_register']:
        idx = e['unique_index']
        delta = abs(probs_t[idx] - probs_a[idx])
        max_delta = max(max_delta, delta)
        print(f"  |{e['binary']}>  {probs_t[idx]:9.6f}  {probs_a[idx]:9.6f}  {delta:8.6f}  {e['codon']}")
    print(f"\n  Max |Δp|: {max_delta:.6f}")


def aae_noisy_fidelity(step1_result, step2_result, shots=8192):
    """
    Run the trained AAE circuit on both ideal (Aer) and noisy (FakeQuebec)
    backends. Computes fidelity via two methods:

      A) Density matrix based (exact noisy probabilities, no shot noise)
         - State fidelity: F = |<psi_target|rho_noisy|psi_target>|
         - Hellinger from DM diagonal: F_H(p_target, diag(rho_noisy))

      B) Shot-count based (finite-sample, subject to shot noise)
         - Hellinger from shots: F_H(p_target, p_shots)

    Returns dict with all metrics + a comparison table.
    """
    import time
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeQuebec
    from qiskit.quantum_info import state_fidelity

    n_q = step2_result['num_qubits']
    target = step1_result['d_normalized']
    target_probs = target ** 2
    circuit = step2_result['circuit']
    n_states = 2 ** n_q

    # Add measurements
    circuit_meas = QuantumCircuit(n_q, n_q)
    circuit_meas.compose(circuit, inplace=True)
    circuit_meas.measure(range(n_q), range(n_q))

    # --- Ideal statevector (already computed) ---
    trained_sv = np.array(step2_result['initial_sv'].data)
    ideal_probs = np.abs(trained_sv) ** 2
    ideal_overlap = step2_result['overlap']
    ideal_fidelity = ideal_overlap ** 2
    target_dm = step2_result['target_dm']
    ideal_dm = step2_result['initial_dm']

    # --- Ideal Aer shots ---
    aer_sim = AerSimulator()
    transpiled_aer = transpile(circuit_meas, backend=aer_sim, optimization_level=1)
    aer_counts = aer_sim.run(transpiled_aer, shots=shots).result().get_counts()

    aer_probs_shots = np.zeros(n_states)
    total_aer = sum(aer_counts.values())
    for bitstring, count in aer_counts.items():
        idx = int(bitstring[::-1], 2)
        aer_probs_shots[idx] = count / total_aer

    # --- Noisy FakeQuebec shots ---
    print(f"  Running noisy AAE on FakeQuebec ({shots} shots)...")
    fake_backend = FakeQuebec()

    t0 = time.time()
    transpiled_quebec = transpile(circuit_meas, backend=fake_backend, optimization_level=3)
    transpile_time = time.time() - t0

    gc_transpiled = dict(transpiled_quebec.count_ops())
    two_q_gates = sum(v for k, v in gc_transpiled.items()
                      if k in ['cx', 'cnot', 'ecr', 'cz', 'swap', 'iswap'])
    transpiled_depth = transpiled_quebec.depth()
    transpiled_total = sum(gc_transpiled.values())
    transpiled_swaps = gc_transpiled.get('swap', 0)

    t0 = time.time()
    quebec_counts = fake_backend.run(transpiled_quebec, shots=shots).result().get_counts()
    run_time = time.time() - t0

    noisy_probs_shots = np.zeros(n_states)
    total_noisy = sum(quebec_counts.values())
    for bitstring, count in quebec_counts.items():
        idx = int(bitstring[::-1], 2)
        noisy_probs_shots[idx] = count / total_noisy

    # --- Noisy density matrix (exact noisy probabilities + state fidelity) ---
    print(f"  Computing noisy density matrix...")
    noisy_probs_dm = None
    has_dm = False
    sf_target_ideal = sf_target_noisy = sf_ideal_noisy = sf_noise_drop = None
    hf_target_noisy_dm = hf_ideal_noisy_dm = None
    tv_noisy_dm = None

    try:
        noise_model = NoiseModel.from_backend(fake_backend)
        dm_sim = AerSimulator(method='density_matrix', noise_model=noise_model)
        qc_dm = circuit.copy()
        qc_dm.save_density_matrix()
        transpiled_dm = transpile(qc_dm, backend=dm_sim, optimization_level=3)
        dm_result = dm_sim.run(transpiled_dm).result()
        noisy_dm = DensityMatrix(dm_result.data()['density_matrix'])

        # Exact noisy probabilities from density matrix diagonal
        noisy_probs_dm = np.real(np.diag(noisy_dm.data))
        noisy_probs_dm = np.clip(noisy_probs_dm, 0, None)

        # State fidelity (gold standard)
        sf_target_ideal = float(state_fidelity(target_dm, ideal_dm))
        sf_target_noisy = float(state_fidelity(target_dm, noisy_dm))
        sf_ideal_noisy = float(state_fidelity(ideal_dm, noisy_dm))
        sf_noise_drop = sf_target_ideal - sf_target_noisy
        has_dm = True
    except Exception as e:
        print(f"  Warning: density matrix simulation failed: {e}")

    # --- Hellinger fidelity ---
    def hellinger_fidelity(p, q):
        return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)))) ** 2

    # From exact probabilities (statevector / DM diagonal)
    hf_target_ideal = hellinger_fidelity(target_probs, ideal_probs)
    if has_dm:
        hf_target_noisy_dm = hellinger_fidelity(target_probs, noisy_probs_dm)
        hf_ideal_noisy_dm = hellinger_fidelity(ideal_probs, noisy_probs_dm)
        tv_noisy_dm = 0.5 * float(np.sum(np.abs(target_probs - noisy_probs_dm)))

    # From shot counts (finite-sample)
    hf_target_aer_shots = hellinger_fidelity(target_probs, aer_probs_shots)
    hf_target_noisy_shots = hellinger_fidelity(target_probs, noisy_probs_shots)
    hf_ideal_noisy_shots = hellinger_fidelity(ideal_probs, noisy_probs_shots)

    # --- TV distance ---
    tv_ideal = 0.5 * float(np.sum(np.abs(target_probs - ideal_probs)))
    tv_noisy_shots = 0.5 * float(np.sum(np.abs(target_probs - noisy_probs_shots)))

    # --- Leakage ---
    codon_indices = np.where(target_probs > 1e-12)[0]
    leakage_aer = 1.0 - float(np.sum(aer_probs_shots[codon_indices]))
    leakage_noisy = 1.0 - float(np.sum(noisy_probs_shots[codon_indices]))

    # --- Print report ---
    print(f"  Transpiled for Quebec: depth={transpiled_depth}, gates={transpiled_total}, 2Q={two_q_gates}, SWAPs={transpiled_swaps}")
    print(f"  Noisy shot simulation done in {run_time:.1f}s")

    print(f"\n" + "=" * 78)
    print(f"  FIDELITY COMPARISON TABLE")
    print(f"=" * 78)
    print(f"\n  {'Metric':<42} {'Ideal':>10} {'Quebec':>10} {'Drop':>10}")
    print(f"  {'-'*42} {'-'*10} {'-'*10} {'-'*10}")

    if has_dm:
        print(f"  {'State fidelity (density matrix)':<42} {sf_target_ideal:>10.6f} {sf_target_noisy:>10.6f} {sf_noise_drop:>10.6f}")
        print(f"  {'Hellinger fidelity (exact probs/DM)':<42} {hf_target_ideal:>10.6f} {hf_target_noisy_dm:>10.6f} {hf_target_ideal - hf_target_noisy_dm:>10.6f}")

    print(f"  {'Hellinger fidelity (shots, n={shots})':<42} {hf_target_aer_shots:>10.6f} {hf_target_noisy_shots:>10.6f} {hf_target_aer_shots - hf_target_noisy_shots:>10.6f}")

    tv_noisy_display = tv_noisy_dm if tv_noisy_dm is not None else tv_noisy_shots
    print(f"  {'TV distance':<42} {tv_ideal:>10.6f} {tv_noisy_display:>10.6f} {'':>10}")
    print(f"  {'Leakage (%)':<42} {100*leakage_aer:>9.2f}% {100*leakage_noisy:>9.2f}% {'':>10}")

    if has_dm:
        print(f"\n  Cross-fidelity (ideal vs noisy circuit):")
        print(f"    F_state(ideal, noisy):          {sf_ideal_noisy:.6f}")
        print(f"    H_F(ideal, noisy) [DM]:         {hf_ideal_noisy_dm:.6f}")
        print(f"    H_F(ideal, noisy) [shots]:      {hf_ideal_noisy_shots:.6f}")

    print(f"\n  Note: Hellinger from DM uses exact noisy probabilities (no sampling)")
    print(f"  while Hellinger from shots uses {shots} samples across {n_states} bins")
    print(f"  (~{shots//n_states} shots/bin). Low-prob codons often get 0 counts,")
    print(f"  zeroing their sqrt(p*q) contribution and depressing shot-based fidelity.")

    return {
        'ideal_overlap': float(ideal_overlap),
        'ideal_fidelity': float(ideal_fidelity),
        # State fidelity (density matrix)
        'sf_target_ideal': sf_target_ideal,
        'sf_target_noisy': sf_target_noisy,
        'sf_ideal_noisy': sf_ideal_noisy,
        'sf_noise_drop': sf_noise_drop,
        # Hellinger from exact probs (DM diagonal)
        'hf_target_ideal': float(hf_target_ideal),
        'hf_target_noisy_dm': float(hf_target_noisy_dm) if hf_target_noisy_dm is not None else None,
        'hf_ideal_noisy_dm': float(hf_ideal_noisy_dm) if hf_ideal_noisy_dm is not None else None,
        # Hellinger from shot counts
        'hf_target_aer_shots': float(hf_target_aer_shots),
        'hf_target_noisy_shots': float(hf_target_noisy_shots),
        'hf_ideal_noisy_shots': float(hf_ideal_noisy_shots),
        # TV distance
        'tv_ideal': float(tv_ideal),
        'tv_noisy_dm': float(tv_noisy_dm) if tv_noisy_dm is not None else None,
        'tv_noisy_shots': float(tv_noisy_shots),
        # Leakage
        'leakage_aer': float(leakage_aer),
        'leakage_noisy': float(leakage_noisy),
        # Circuit metrics
        'transpiled_depth': int(transpiled_depth),
        'transpiled_total_gates': int(transpiled_total),
        'transpiled_2q_gates': int(two_q_gates),
        'transpiled_swaps': int(transpiled_swaps),
        'transpile_time_s': float(transpile_time),
        'run_time_s': float(run_time),
        'shots': shots,
    }


# =====================================================================
# CACHED-PARAMS WORKFLOW
# =====================================================================
#
# Pattern: train AAE once (classical, slow), dump the trained RY angles
# to JSON, then in downstream QSP/QSVT pipelines just reload the JSON and
# rebuild the brickwall circuit. Reloading is milliseconds — the heavy
# L-BFGS-B optimization only runs when you actually want to re-train
# (e.g. trying a different number of layers).
#
# Schema (results/best_aae_params_gapdh.json):
#   {
#     "params":    [float, ...],   # length = n_qubits * n_layers
#     "cost":      float,          # 1 - Re<target|sv> from training
#     "overlap":   float,          # |<target|sv>|
#     "n_qubits":  int,
#     "n_layers":  int,
#     "dataset":   str,            # tag for audit, e.g. "GAPDH_4species"
#     "timestamp": str,            # UTC ISO-8601, set at save time
#   }
# Older files without `dataset` / `timestamp` are still accepted by the
# loader; those fields are optional.


def save_aae_params(json_path, aae_result, dataset_tag=None):
    """
    Dump a trained AAE record to JSON in the canonical schema.

    Accepts the dict returned by either aae_encode() or load_aae_circuit().
    Creates the parent directory if needed.
    """
    payload = {
        'params'   : np.asarray(aae_result['best_params']).tolist(),
        'cost'     : float(aae_result['best_cost']),
        'overlap'  : float(aae_result['overlap']),
        'n_qubits' : int(aae_result['num_qubits']),
        'n_layers' : int(aae_result['n_layers']),
        'dataset'  : dataset_tag or 'unknown',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    out_dir = os.path.dirname(os.path.abspath(json_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_aae_circuit(json_path, step1_result,
                     expected_n_qubits=None, expected_n_layers=None):
    """
    Load a previously-trained AAE circuit from a JSON file.

    Returns a dict in the same shape as aae_encode()'s output, so
    downstream pipelines (QSP/QSVT) work without code changes.

    The statevector is recomputed from the rebuilt circuit (cheap, ms).
    Cost/overlap are taken from the JSON when present, else recomputed
    against step1_result['d_normalized'].

    Parameters
    ----------
    json_path : str
        Path to a JSON file produced by save_aae_params() or
        scripts/aae_results_gapdh.py.
    step1_result : dict
        Output of data.gapdh_sequences.build_gapdh_register(). Needed
        for the target state used in fidelity bookkeeping.
    expected_n_qubits, expected_n_layers : int, optional
        If provided, raise ValueError on mismatch with the JSON.

    Raises
    ------
    FileNotFoundError
        If json_path does not exist.
    ValueError
        On schema / shape / consistency mismatches.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"AAE params file not found: {json_path}\n"
            f"  Train and dump with: python scripts/aae_results_gapdh.py")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Required fields
    for key in ('params', 'n_qubits', 'n_layers'):
        if key not in data:
            raise ValueError(f"AAE params file missing required field '{key}': {json_path}")

    n_q      = int(data['n_qubits'])
    n_layers = int(data['n_layers'])
    params   = np.asarray(data['params'], dtype=float)

    expected_n_params = n_q * n_layers
    if len(params) != expected_n_params:
        raise ValueError(
            f"AAE params file shape mismatch: len(params)={len(params)} "
            f"but n_qubits*n_layers={expected_n_params} ({json_path})")

    if n_q != step1_result['num_qubits']:
        raise ValueError(
            f"AAE params n_qubits ({n_q}) != step1_result num_qubits "
            f"({step1_result['num_qubits']}). Re-train for the right register size.")

    if expected_n_qubits is not None and n_q != expected_n_qubits:
        raise ValueError(f"AAE params n_qubits ({n_q}) != expected ({expected_n_qubits}).")
    if expected_n_layers is not None and n_layers != expected_n_layers:
        raise ValueError(f"AAE params n_layers ({n_layers}) != expected ({expected_n_layers}).")

    # Rebuild the trained circuit — RY angles baked in as concrete floats.
    d = step1_result['d_normalized']
    circuit = build_brickwall_ansatz(n_q, n_layers, params)

    circuit_meas = QuantumCircuit(n_q, n_q)
    circuit_meas.compose(circuit, inplace=True)
    circuit_meas.measure(range(n_q), range(n_q))

    trained_sv = Statevector.from_instruction(circuit)
    sv_data    = np.asarray(trained_sv.data)

    overlap = float(data['overlap']) if 'overlap' in data else float(abs(np.vdot(d, sv_data)))
    cost    = float(data['cost'])    if 'cost'    in data else float(1.0 - np.real(np.vdot(d, sv_data)))

    gc = dict(circuit.count_ops())

    return {
        'encoding_type' : 'aae_loaded',
        'circuit'       : circuit,
        'circuit_meas'  : circuit_meas,
        'initial_sv'    : trained_sv,
        'initial_dm'    : DensityMatrix(trained_sv),
        'target_sv'     : Statevector(d),
        'target_dm'     : DensityMatrix(Statevector(d)),
        'num_qubits'    : n_q,
        'n_layers'      : n_layers,
        'best_params'   : params,
        'best_cost'     : cost,
        'overlap'       : overlap,
        'logical_cnot'  : gc.get('cx', 0),
        'logical_ry'    : gc.get('ry', 0),
        'logical_total' : gc.get('cx', 0) + gc.get('ry', 0),
        'source_json'   : os.path.abspath(json_path),
        'dataset'       : data.get('dataset'),
        'timestamp'     : data.get('timestamp'),
    }


def get_aae_circuit(step1_result, json_path,
                   n_layers=6, n_trials=6, maxiter=5000,
                   force_retrain=False, dataset_tag='GAPDH_4species',
                   save_after_train=True):
    """
    Recommended entry point for QSP/QSVT pipelines.

    Behavior:
      - json_path exists and force_retrain=False → load and reuse it
        AS-IS in milliseconds, regardless of the n_layers argument.
        The cache is treated as authoritative when present.
      - Cache missing OR force_retrain=True → run aae_encode() with
        the supplied n_layers/n_trials/maxiter and save to json_path.

    The pipelines should ALWAYS call this rather than aae_encode()
    directly. Re-training is strictly opt-in:
        get_aae_circuit(...)                       # use cache if present
        get_aae_circuit(..., force_retrain=True)   # force fresh training

    The n_layers / n_trials / maxiter arguments only take effect when
    training is actually triggered. To switch caches without retraining,
    point json_path at a different filename. If a cache exists but is
    structurally broken (wrong n_qubits, missing fields), load_aae_circuit
    raises ValueError and this function falls back to retraining.
    """
    if not force_retrain and os.path.isfile(json_path):
        try:
            cached = load_aae_circuit(json_path, step1_result)
        except ValueError as e:
            print(f"  Cache at {json_path} is incompatible: {e}")
            print(f"  Re-training fresh.")
            cached = None

        if cached is not None:
            print(f"  Loaded cached AAE params from {json_path}")
            print(f"    n_qubits={cached['num_qubits']}  n_layers={cached['n_layers']}  "
                  f"overlap={cached['overlap']:.6f}  cost={cached['best_cost']:.6f}")
            if cached.get('timestamp'):
                print(f"    Trained: {cached['timestamp']} "
                      f"({cached.get('dataset', 'unknown')})")
            if cached['n_layers'] != n_layers:
                print(f"    Note: cache has n_layers={cached['n_layers']}, "
                      f"caller default was {n_layers} — honoring cache.")
            return cached

    if force_retrain and os.path.isfile(json_path):
        print(f"  force_retrain=True — retraining and overwriting {json_path}")
    elif not os.path.isfile(json_path):
        print(f"  No cached params at {json_path} — training fresh.")
        print(f"  (Trained params will be saved for reuse.)")

    result = aae_encode(step1_result, n_layers=n_layers,
                        n_trials=n_trials, maxiter=maxiter)

    if save_after_train:
        save_aae_params(json_path, result, dataset_tag=dataset_tag)
        print(f"  Saved trained AAE params to {json_path}")

    return result
