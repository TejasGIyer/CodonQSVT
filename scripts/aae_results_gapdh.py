"""
AAE Results for Paper -- Section II: Quantum State Encoding
============================================================
Runs the full AAE pipeline on the GAPDH 4-species data and saves
all metrics needed for the paper.

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/aae_results_gapdh.py
"""

import os, sys, json, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from data.gapdh_sequences import (build_gapdh_register, pooled_codon_frequencies,
                                   SENSE_CODONS_SORTED, ALL_SEQUENCES)
from src.aae_encoding import (aae_encode, build_brickwall_ansatz, print_step2,
                              aae_noisy_fidelity, save_aae_params)
from src.gy94_model import GENETIC_CODE
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train AAE on GAPDH codon distribution and dump trained params to JSON.")
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Brickwall layers (default: 6).')
    parser.add_argument('--n-trials', type=int, default=6,
                        help='Random L-BFGS-B restarts (default: 6).')
    parser.add_argument('--maxiter', type=int, default=5000,
                        help='Max L-BFGS-B iterations per trial (default: 5000).')
    parser.add_argument('--out', type=str,
                        default=os.path.join(RESULTS_DIR, 'best_aae_params_gapdh.json'),
                        help='JSON output path for trained params.')
    parser.add_argument('--skip-noisy', action='store_true',
                        help='Skip the FakeQuebec noisy fidelity step (faster iteration).')
    args = parser.parse_args()

    print("=" * 70)
    print("  AAE RESULTS FOR PAPER -- GAPDH 4-Species")
    print("=" * 70)

    # === 1. Build target state ===
    reg = build_gapdh_register(n_qubits=6)
    target = reg['d_normalized']
    target_probs = target ** 2
    n_q = reg['num_qubits']
    freqs = pooled_codon_frequencies()
    n_unique = reg['num_unique']

    print(f"\n  Dataset: GAPDH (Human, Mouse, Rat, Dog)")
    print(f"  Sequences: {len(ALL_SEQUENCES)} species x 334 codons = {reg['num_codons']} total codons")
    print(f"  Unique sense codons observed: {n_unique}")
    print(f"  Qubits: {n_q}  (2^{n_q} = {2**n_q} states, {n_unique} used)")

    # === 2. Train AAE ===
    N_LAYERS, N_TRIALS, MAXITER = args.n_layers, args.n_trials, args.maxiter
    n_params = n_q * N_LAYERS
    print(f"\n  Training AAE: {N_LAYERS} layers, {N_TRIALS} trials, {MAXITER} max iterations")
    print(f"  Parameters: {n_params}")

    t0 = time.time()
    s2 = aae_encode(reg, n_layers=N_LAYERS, n_trials=N_TRIALS, maxiter=MAXITER)
    train_time = time.time() - t0
    print(f"\n  Training complete in {train_time:.1f}s")

    # === Save trained params right after training, before any noisy work.
    # This way --skip-noisy iteration loops are fast, and the JSON is up to
    # date even if the noisy/transpile section crashes for unrelated reasons.
    save_aae_params(args.out, s2, dataset_tag='GAPDH_4species')
    print(f"  Trained params saved to: {args.out}")

    if args.skip_noisy:
        print(f"\n  --skip-noisy: exiting before noisy evaluation. Overlap = {s2['overlap']:.6f}")
        return

    # === 3. Extract metrics ===
    trained_sv = np.array(s2['initial_sv'].data)
    achieved_probs = np.abs(trained_sv) ** 2
    overlap = s2['overlap']
    fidelity = overlap ** 2
    cost = s2['best_cost']

    circuit = s2['circuit']
    lc = dict(circuit.count_ops())
    n_cx = lc.get('cx', 0)
    n_ry = lc.get('ry', 0)
    depth = circuit.depth()

    tv_distance = 0.5 * np.sum(np.abs(target_probs - achieved_probs[:len(target_probs)]))
    max_delta = max(abs(target_probs[e['unique_index']] - achieved_probs[e['unique_index']])
                    for e in reg['unique_register'])

    # === 3b. Noisy fidelity (FakeQuebec) ===
    print(f"\n  Running noisy fidelity evaluation...")
    noisy = aae_noisy_fidelity(reg, s2, shots=8192)

    # === 4. Print report ===
    print(f"\n" + "=" * 70)
    print(f"  AAE ENCODING RESULTS")
    print(f"=" * 70)
    print(f"\n  Encoding: AAE (Brickwall, L-BFGS-B, best of {N_TRIALS} restarts)")
    print(f"\n  Logical circuit: {n_q} qubits, {N_LAYERS} layers, {n_params} params")
    print(f"    {n_ry} Ry + {n_cx} CX = {n_ry+n_cx} gates, depth {depth}")
    print(f"\n  Performance (ideal):")
    print(f"    Overlap: {overlap:.6f}   Fidelity: {fidelity:.6f}")
    print(f"    TV distance: {tv_distance:.6f}   Max |dp|: {max_delta:.6f}")
    print(f"    Training time: {train_time:.1f}s")

    print(f"\n  Performance (FakeQuebec, {noisy['shots']} shots):")
    if noisy.get('sf_target_noisy') is not None:
        print(f"    State F(target,ideal):  {noisy['sf_target_ideal']:.6f}")
        print(f"    State F(target,noisy):  {noisy['sf_target_noisy']:.6f}")
        print(f"    State F(ideal,noisy):   {noisy['sf_ideal_noisy']:.6f}")
        print(f"    Noise drop (state):     {noisy['sf_noise_drop']:.6f}")
    if noisy.get('hf_target_noisy_dm') is not None:
        print(f"    Hellinger (exact/DM):   {noisy['hf_target_noisy_dm']:.6f}")
        print(f"    Hellinger (shots):      {noisy['hf_target_noisy_shots']:.6f}")
    print(f"    Transpiled: depth={noisy['transpiled_depth']}, 2Q={noisy['transpiled_2q_gates']}, SWAPs={noisy['transpiled_swaps']}")

    # Per-codon comparison
    entries = []
    for entry in reg['unique_register']:
        idx = entry['unique_index']
        codon = entry['codon']
        aa = GENETIC_CODE.get(codon, '?')
        pt, pa = target_probs[idx], achieved_probs[idx]
        entries.append((codon, aa, pt, pa, abs(pt - pa), idx))
    entries.sort(key=lambda x: -x[2])

    print(f"\n  Per-codon (top 15 + bottom 5):")
    print(f"  {'Codon':>6} {'AA':>4} {'p_target':>10} {'p_achieved':>10} {'|dp|':>10} {'Match':>6}")
    print(f"  {'-'*6} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for c, aa, pt, pa, d, _ in entries[:15]:
        m = "ok" if d < 0.001 else ("~" if d < 0.003 else "x")
        print(f"  {c:>6} {aa:>4} {pt:10.6f} {pa:10.6f} {d:10.6f} {m:>6}")
    print(f"  {'...':>6}")
    for c, aa, pt, pa, d, _ in entries[-5:]:
        m = "ok" if d < 0.001 else ("~" if d < 0.003 else "x")
        print(f"  {c:>6} {aa:>4} {pt:10.6f} {pa:10.6f} {d:10.6f} {m:>6}")

    # === 5. Encoding comparison -- ACTUAL transpilation for all three ===
    from qiskit.circuit.library import StatePreparation
    from qiskit import transpile as qk_transpile
    from qiskit_ibm_runtime.fake_provider import FakeQuebec

    fake_backend = FakeQuebec()

    # Mottonen
    print(f"\n  Transpiling Mottonen (StatePreparation) for FakeQuebec...")
    mqc = QuantumCircuit(n_q)
    mqc.append(StatePreparation(target), range(n_q))
    mt = qk_transpile(mqc, backend=fake_backend, optimization_level=3)
    mgc = dict(mt.count_ops())
    m_depth = mt.depth()
    m_2q = sum(v for k, v in mgc.items() if k in ['cx','cnot','ecr','cz','swap','iswap'])
    m_total = sum(mgc.values())
    m_swaps = mgc.get('swap', 0)
    print(f"    Mottonen: depth={m_depth}, gates={m_total}, 2Q={m_2q}, SWAPs={m_swaps}")

    # Angle
    print(f"  Transpiling angle encoding ({n_unique} qubits) for FakeQuebec...")
    aqc = QuantumCircuit(n_unique)
    for i, (codon, freq) in enumerate(sorted(freqs.items(), key=lambda x: -x[1])):
        if i >= n_unique: break
        aqc.ry(2 * np.arcsin(np.sqrt(freq)), i)
    at = qk_transpile(aqc, backend=fake_backend, optimization_level=3)
    agc = dict(at.count_ops())
    a_depth = at.depth()
    a_2q = sum(v for k, v in agc.items() if k in ['cx','cnot','ecr','cz','swap','iswap'])
    a_total = sum(agc.values())
    a_swaps = agc.get('swap', 0)
    print(f"    Angle: depth={a_depth}, gates={a_total}, 2Q={a_2q}, SWAPs={a_swaps}")

    # AAE (from noisy results)
    aae_d = noisy['transpiled_depth']
    aae_2q = noisy['transpiled_2q_gates']
    aae_t = noisy['transpiled_total_gates']
    aae_sw = noisy['transpiled_swaps']

    quebec_hf = noisy.get('hf_target_noisy_dm') or noisy['hf_target_noisy_shots']

    print(f"\n" + "=" * 78)
    print(f"  ENCODING COMPARISON TABLE (all transpiled for FakeQuebec)")
    print(f"=" * 78)
    print(f"\n  {'Encoding':<25} {'Qubits':>7} {'Depth':>7} {'2Q gates':>9} {'Total':>7} {'SWAPs':>6} {'Fidelity':>9}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*6} {'-'*9}")
    print(f"  {'Amplitude (Mottonen)':<25} {n_q:>7} {m_depth:>7} {m_2q:>9} {m_total:>7} {m_swaps:>6} {'1.000000':>9}")
    print(f"  {'Angle (Ry)':<25} {n_unique:>7} {a_depth:>7} {a_2q:>9} {a_total:>7} {a_swaps:>6} {'1.000000':>9}")
    print(f"  {'AAE (Brickwall, L=6)':<25} {n_q:>7} {aae_d:>7} {aae_2q:>9} {aae_t:>7} {aae_sw:>6} {quebec_hf:>9.6f}")

    # === 6. Save results ===
    results = {
        'dataset': 'GAPDH_4species',
        'species': list(ALL_SEQUENCES.keys()),
        'n_codons_total': reg['num_codons'],
        'n_unique_sense': n_unique,
        'n_qubits': n_q, 'n_layers': N_LAYERS, 'n_params': n_params,
        'n_trials': N_TRIALS, 'maxiter': MAXITER, 'optimizer': 'L-BFGS-B',
        'overlap': float(overlap), 'fidelity': float(fidelity),
        'cost': float(cost), 'tv_distance': float(tv_distance),
        'max_delta_p': float(max_delta), 'train_time_s': float(train_time),
        'circuit_depth': int(depth), 'n_cx_gates': int(n_cx), 'n_ry_gates': int(n_ry),
        'per_codon': [
            {'codon': c, 'amino_acid': aa, 'p_target': float(pt),
             'p_achieved': float(pa), 'delta': float(d)}
            for c, aa, pt, pa, d, _ in entries
        ],
        'encoding_comparison': {
            'mottonen': {'qubits': n_q, 'depth': m_depth, 'two_q_gates': m_2q,
                         'total_gates': m_total, 'swaps': m_swaps, 'fidelity': 1.0},
            'angle':    {'qubits': n_unique, 'depth': a_depth, 'two_q_gates': a_2q,
                         'total_gates': a_total, 'swaps': a_swaps, 'fidelity': 1.0},
            'aae':      {'qubits': n_q, 'depth': aae_d, 'two_q_gates': aae_2q,
                         'total_gates': aae_t, 'swaps': aae_sw, 'fidelity': float(quebec_hf)},
        },
        'noisy_quebec': noisy,
    }

    out_path = os.path.join(RESULTS_DIR, 'aae_results_gapdh.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"  (Trained params were already saved earlier to: {args.out})")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
