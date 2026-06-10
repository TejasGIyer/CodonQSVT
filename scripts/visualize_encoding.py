"""
Visualization for Section II: Quantum State Encoding
======================================================
Generates publication-quality figures for the paper.

Figures:
  1. GAPDH pooled codon frequency histogram (target distribution)
  2. Brickwall ansatz circuit schematic (Qiskit draw)
  3. Target vs achieved probability distributions (grouped bars)
  4. Optimizer convergence curves (cost vs evaluations)
  5. Optimizer comparison bar chart (final overlap)
  6. Qubit scaling plot (sequence length vs qubits needed)

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/visualize_encoding.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from data.gapdh_sequences import (
    build_gapdh_register, pooled_codon_frequencies,
    SENSE_CODONS_SORTED, ALL_SEQUENCES
)
from src.aae_encoding import build_brickwall_ansatz

FIGURES_DIR = os.path.join(_PROJECT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})


# =========================================================================
# FIGURE 1: GAPDH Pooled Codon Frequency Histogram
# =========================================================================

def fig1_codon_frequency_histogram():
    """Bar chart of pooled codon frequencies across 4 GAPDH species."""
    print("  [1/6] Codon frequency histogram...")

    freqs = pooled_codon_frequencies()
    # Sort by frequency descending
    sorted_codons = sorted(freqs.items(), key=lambda x: -x[1])
    codons = [c for c, _ in sorted_codons]
    values = [v for _, v in sorted_codons]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color by amino acid type
    from src.gy94_model import GENETIC_CODE
    aa_colors = {}
    cmap = plt.cm.tab20
    unique_aas = sorted(set(GENETIC_CODE[c] for c in codons if c in GENETIC_CODE))
    for i, aa in enumerate(unique_aas):
        aa_colors[aa] = cmap(i / len(unique_aas))

    colors = [aa_colors.get(GENETIC_CODE.get(c, 'Unk'), '#888888') for c in codons]

    bars = ax.bar(range(len(codons)), values, color=colors, edgecolor='none',
                  width=0.8, alpha=0.85)

    ax.set_xlabel('Codon (sorted by frequency)')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Pooled codon frequency distribution — GAPDH (Human, Mouse, Rat, Dog)')
    ax.set_xticks(range(len(codons)))
    ax.set_xticklabels(codons, rotation=90, fontsize=6.5, fontfamily='monospace')

    # Annotate top 5
    for i in range(min(5, len(codons))):
        aa = GENETIC_CODE.get(codons[i], '?')
        ax.annotate(f'{codons[i]}\n({aa})',
                    xy=(i, values[i]), xytext=(i + 2, values[i] + 0.003),
                    fontsize=7, ha='center',
                    arrowprops=dict(arrowstyle='->', color='#333', lw=0.5))

    ax.set_xlim(-0.5, len(codons) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_codon_frequency_histogram.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 2: Brickwall Ansatz Circuit Schematic
# =========================================================================

def fig2_brickwall_ansatz():
    """Draw the brickwall ansatz circuit using Qiskit's drawer."""
    print("  [2/6] Brickwall ansatz circuit...")

    n_qubits = 6
    n_layers = 6
    params = np.zeros(n_qubits * n_layers)  # placeholder angles
    # Label params as theta
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name='AAE Brickwall')
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(Parameter(f'θ_{{{idx}}}'), q)
            idx += 1
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        if layer < n_layers - 1:
            qc.barrier(label=f'L{layer+1}')

    fig = qc.draw(output='mpl', style={'fontsize': 8, 'subfontsize': 6},
                  fold=80, scale=0.7)
    path = os.path.join(FIGURES_DIR, 'fig2_brickwall_ansatz.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 3: Target vs Achieved Probability Distributions
# =========================================================================

def fig3_target_vs_achieved():
    """Grouped bar chart: target vs AAE-trained probability distributions."""
    print("  [3/6] Target vs achieved distributions...")

    reg = build_gapdh_register(n_qubits=6)
    target = reg['d_normalized']
    target_probs = target ** 2

    # Check if we have saved best params
    params_path = os.path.join(_SCRIPT_DIR, 'results', 'best_aae_params_gapdh.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            saved = json.load(f)
        params = np.array(saved['params'])
        n_q, n_l = saved['n_qubits'], saved['n_layers']
    else:
        # Run quick training
        print("    No saved params found, running quick AAE training...")
        from src.aae_encoding import aae_encode
        s2 = aae_encode(reg, n_layers=6, n_trials=2, maxiter=1000)
        params = s2['best_params']
        n_q, n_l = s2['num_qubits'], s2['n_layers']

    from qiskit.quantum_info import Statevector
    qc = build_brickwall_ansatz(n_q, n_l, params)
    sv = np.array(Statevector.from_instruction(qc).data)
    achieved_probs = np.abs(sv) ** 2

    # Get nonzero codons
    nonzero_indices = np.where(target_probs > 1e-6)[0]
    n_show = len(nonzero_indices)

    # Get codon labels
    codon_labels = []
    for idx in nonzero_indices:
        found = False
        for entry in reg['unique_register']:
            if entry['unique_index'] == idx:
                codon_labels.append(entry['codon'])
                found = True
                break
        if not found:
            codon_labels.append(f'|{idx}⟩')

    x = np.arange(n_show)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width/2, target_probs[nonzero_indices], width,
                   label='Target (exact)', color='#3498db', alpha=0.85, edgecolor='none')
    bars2 = ax.bar(x + width/2, achieved_probs[nonzero_indices], width,
                   label='AAE trained', color='#e74c3c', alpha=0.85, edgecolor='none')

    ax.set_xlabel('Codon')
    ax.set_ylabel('Probability')
    ax.set_title('Target vs AAE-trained codon probability distribution (6 qubits, 6 layers)')
    ax.set_xticks(x)
    ax.set_xticklabels(codon_labels, rotation=90, fontsize=7, fontfamily='monospace')
    ax.legend(framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Compute and annotate overlap
    overlap = np.abs(np.vdot(target, sv))
    ax.text(0.98, 0.95, f'Overlap = {overlap:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_target_vs_achieved.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 4: Optimizer Convergence Curves
# =========================================================================

def fig4_optimizer_convergence():
    """Plot cost vs evaluations for all 5 optimizers."""
    print("  [4/6] Optimizer convergence curves...")

    bench_path = os.path.join(_SCRIPT_DIR, 'results', 'optimizer_benchmark_gapdh.json')
    if not os.path.exists(bench_path):
        print(f"    Benchmark data not found at {bench_path}")
        print("    Run optimizer_benchmark_gapdh.py first!")
        return

    with open(bench_path) as f:
        results = json.load(f)

    colors = {
        'L-BFGS': '#2ecc71',
        'COBYLA': '#3498db',
        'Nelder-Mead': '#9b59b6',
        'SPSA': '#e74c3c',
        'Adam+ParamShift': '#f39c12',
    }
    markers = {
        'L-BFGS': 'o',
        'COBYLA': 's',
        'Nelder-Mead': '^',
        'SPSA': 'D',
        'Adam+ParamShift': 'v',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: cost vs evaluations (log scale)
    for r in results:
        name = r['name']
        hist = r['history']
        if not hist:
            continue
        evals = [h['eval'] for h in hist]
        costs = [h['cost'] for h in hist]
        ax1.plot(evals, costs, color=colors.get(name, '#888'),
                 marker=markers.get(name, '.'), markersize=3,
                 linewidth=1.5, label=name, alpha=0.85)

    ax1.set_xlabel('Cost function evaluations')
    ax1.set_ylabel('Cost (1 - overlap)')
    ax1.set_title('Optimizer convergence (cost vs evaluations)')
    ax1.set_yscale('log')
    ax1.legend(framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(bottom=1e-3)

    # Right: cost vs wall-clock time
    for r in results:
        name = r['name']
        hist = r['history']
        if not hist or len(hist) < 2:
            continue
        total_time = r['time']
        total_evals = r['evals']
        times = [(h['eval'] / total_evals) * total_time for h in hist]
        costs = [h['cost'] for h in hist]
        ax2.plot(times, costs, color=colors.get(name, '#888'),
                 marker=markers.get(name, '.'), markersize=3,
                 linewidth=1.5, label=name, alpha=0.85)

    ax2.set_xlabel('Wall-clock time (s)')
    ax2.set_ylabel('Cost (1 - overlap)')
    ax2.set_title('Optimizer convergence (cost vs time)')
    ax2.set_yscale('log')
    ax2.legend(framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(bottom=1e-3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_optimizer_convergence.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 5: Optimizer Comparison Bar Chart
# =========================================================================

def fig5_optimizer_comparison():
    """Bar chart comparing final overlap across optimizers."""
    print("  [5/6] Optimizer comparison bar chart...")

    bench_path = os.path.join(_SCRIPT_DIR, 'results', 'optimizer_benchmark_gapdh.json')
    if not os.path.exists(bench_path):
        print(f"    Benchmark data not found. Run optimizer_benchmark_gapdh.py first!")
        return

    with open(bench_path) as f:
        results = json.load(f)

    names = [r['name'] for r in results]
    overlaps = [r['overlap'] for r in results]
    times = [r['time'] for r in results]
    evals = [r['evals'] for r in results]

    colors_bar = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overlap
    bars = ax1.bar(names, overlaps, color=colors_bar, edgecolor='none', alpha=0.85)
    ax1.set_ylabel('Overlap with target state')
    ax1.set_title('Final overlap by optimizer')
    ax1.set_ylim(0.5, 1.0)
    for bar, ov in zip(bars, overlaps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{ov:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', rotation=20)

    # Right: time and evals (dual axis)
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax2.bar(x - width/2, times, width, label='Time (s)',
                    color='#2c3e50', alpha=0.75)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, evals, width, label='Evaluations',
                         color='#e67e22', alpha=0.75)
    ax2.set_ylabel('Time (seconds)')
    ax2_twin.set_ylabel('Cost evaluations')
    ax2.set_title('Computational cost by optimizer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha='right')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2_twin.legend(loc='upper right', framealpha=0.9)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_optimizer_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 6: Qubit Scaling Plot
# =========================================================================

def fig6_qubit_scaling():
    """Plot showing constant qubit scaling with sequence length."""
    print("  [6/6] Qubit scaling plot...")

    from collections import Counter

    # Simulate different sequence lengths
    combined = "".join(ALL_SEQUENCES.values())
    total_len = len(combined)

    # Sample subsequences of various lengths
    seq_lengths = [50, 100, 200, 500, 1000, 2000, 3000, 4008]
    naive_qubits = []  # 2 bits per base
    codon_qubits_amp = []  # log2(unique codons) for amplitude encoding
    codon_qubits_angle = []  # one qubit per unique codon for angle encoding
    n_unique_codons = []

    stop_codons = {'TAA', 'TAG', 'TGA'}

    for L in seq_lengths:
        seq = combined[:L]

        # Naive: 2 bits per base
        naive_qubits.append(2 * L)

        # Codon-based
        codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
        sense = [c for c in codons if len(c) == 3 and c not in stop_codons]
        unique = len(set(sense))
        n_unique_codons.append(unique)

        # Amplitude encoding: ceil(log2(unique))
        n_q_amp = max(1, int(np.ceil(np.log2(max(unique, 2)))))
        codon_qubits_amp.append(n_q_amp)

        # Angle encoding: one qubit per unique codon
        codon_qubits_angle.append(unique)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: qubit count vs sequence length
    ax1.plot(seq_lengths, naive_qubits, 'o-', color='#e74c3c', linewidth=2,
             markersize=6, label='Naive (2 bits/base)', alpha=0.85)
    ax1.plot(seq_lengths, codon_qubits_angle, 's-', color='#f39c12', linewidth=2,
             markersize=6, label='Angle encoding (1 qubit/codon)', alpha=0.85)
    ax1.plot(seq_lengths, codon_qubits_amp, 'D-', color='#2ecc71', linewidth=2,
             markersize=6, label='Amplitude/AAE (⌈log₂ unique⌉)', alpha=0.85)

    ax1.set_xlabel('Sequence length (nucleotides)')
    ax1.set_ylabel('Qubits required')
    ax1.set_title('Qubit scaling by encoding strategy')
    ax1.set_yscale('log')
    ax1.legend(framealpha=0.9, loc='center left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add the 6-qubit ceiling annotation
    ax1.axhline(y=6, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(seq_lengths[-1] * 0.6, 7.5,
             '6-qubit ceiling\n(61 sense codons)',
             fontsize=9, color='#27ae60', ha='center')

    # Right: unique codons vs sequence length (showing saturation)
    ax2.plot(seq_lengths, n_unique_codons, 'o-', color='#3498db', linewidth=2,
             markersize=6, alpha=0.85)
    ax2.axhline(y=61, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.5,
                label='Maximum (61 sense codons)')
    ax2.fill_between([0, seq_lengths[-1] * 1.1], 61, 65, alpha=0.1, color='#e74c3c')

    ax2.set_xlabel('Sequence length (nucleotides)')
    ax2.set_ylabel('Unique sense codons observed')
    ax2.set_title('Codon diversity saturation')
    ax2.legend(framealpha=0.9)
    ax2.set_ylim(0, 70)
    ax2.set_xlim(0, seq_lengths[-1] * 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_qubit_scaling.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  GENERATING FIGURES — Section II: Quantum State Encoding")
    print("=" * 70)

    fig1_codon_frequency_histogram()
    fig2_brickwall_ansatz()
    fig3_target_vs_achieved()
    fig4_optimizer_convergence()
    fig5_optimizer_comparison()
    fig6_qubit_scaling()

    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
