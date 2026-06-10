"""
Publication Figures for Section II: Quantum State Encoding
============================================================
Generates 3 publication-quality figures:

  Fig 1: Encoding comparison 2x2 grid (qubits, depth, CX gates, fidelity)
  Fig 2: Target vs achieved codon probability distribution
  Fig 3: Fidelity comparison (ideal vs Quebec across metrics)

Reads results from results/aae_results_gapdh.json (run aae_results_gapdh.py first).

Usage:
    cd "C:\\Users\\HPUSER\\Desktop\\Genetic Mutation"
    python scripts/paper_figures.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

FIGURES_DIR = os.path.join(_PROJECT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
})

# Consistent colors
C_AMP = '#C0392B'   # Amplitude - red
C_ANG = '#2471A3'   # Angle - blue
C_AAE = '#1E8449'   # AAE - green
C_AAE_NOISY = '#145A32'  # AAE noisy - dark green


def load_results():
    """Load results from aae_results_gapdh.json."""
    path = os.path.join(_PROJECT_DIR, 'results', 'aae_results_gapdh.json')
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found. Run aae_results_gapdh.py first!")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# =========================================================================
# FIGURE 1: Encoding Comparison 2x2 Grid
# =========================================================================

def fig1_encoding_comparison_grid(results):
    """2x2 bar chart: qubits, depth, CX gates, fidelity."""
    print("  [1/3] Encoding comparison 2x2 grid...")

    enc = results['encoding_comparison']
    noisy = results['noisy_quebec']

    # Data
    labels = ['Amplitude', 'Angle', 'AAE']
    colors = [C_AMP, C_ANG, C_AAE]

    qubits = [enc['mottonen']['qubits'], enc['angle']['qubits'], enc['aae']['qubits']]
    depths = [enc['mottonen']['depth'], enc['angle']['depth'], enc['aae']['depth']]
    two_q = [enc['mottonen']['two_q_gates'], enc['angle']['two_q_gates'], enc['aae']['two_q_gates']]

    # For fidelity panel: 4 bars
    fid_labels = ['Amplitude', 'Angle', 'AAE\n(ideal)', 'AAE\n(Quebec)']
    fid_values = [
        enc['mottonen']['fidelity'],
        enc['angle']['fidelity'],
        results['fidelity'],
        enc['aae']['fidelity'],
    ]
    fid_colors = [C_AMP, C_ANG, C_AAE, C_AAE_NOISY]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    panels = [
        (axes[0, 0], '(a)  Qubits', labels, qubits, colors, None),
        (axes[0, 1], '(b)  Transpiled depth', labels, depths, colors, None),
        (axes[1, 0], '(c)  2Q gates', labels, two_q, colors, None),
        (axes[1, 1], '(d)  Fidelity', fid_labels, fid_values, fid_colors, None),
    ]

    for ax, title, xlabels, values, cols, ylim in panels:
        bars = ax.bar(xlabels, values, color=cols, edgecolor='none', width=0.55, alpha=0.88)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

        # Value labels on top of bars
        for bar, val in zip(bars, values):
            if val == 0:
                label = '0'
            elif val < 1 and val > 0:
                label = f'{val:.3f}'
            elif val == int(val):
                label = f'{int(val)}'
            else:
                label = f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Extra headroom for labels
        if max(values) > 0:
            ax.set_ylim(0, max(values) * 1.18)

    # Special y-axis for fidelity panel
    axes[1, 1].set_ylim(0.8, 1.05)

    plt.tight_layout(pad=2.0)
    path = os.path.join(FIGURES_DIR, 'fig1_encoding_comparison_2x2.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 2: Target vs Achieved Codon Distribution
# =========================================================================

def fig2_target_vs_achieved(results):
    """Grouped bar chart: target vs AAE-trained probability for all codons."""
    print("  [2/3] Target vs achieved distribution...")

    per_codon = results['per_codon']
    # Already sorted by p_target descending in the JSON
    codons = [c['codon'] for c in per_codon]
    p_target = [c['p_target'] for c in per_codon]
    p_achieved = [c['p_achieved'] for c in per_codon]

    n = len(codons)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))

    bars1 = ax.bar(x - width / 2, p_target, width,
                   label='Target (exact)', color=C_ANG, alpha=0.85, edgecolor='none')
    bars2 = ax.bar(x + width / 2, p_achieved, width,
                   label='AAE trained', color=C_AAE, alpha=0.85, edgecolor='none')

    ax.set_xlabel('Codon (sorted by target frequency)')
    ax.set_ylabel('Probability')
    ax.set_title('Target vs AAE-trained codon probability distribution — GAPDH (4 species)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(codons, rotation=90, fontsize=6.5, fontfamily='monospace')
    ax.legend(framealpha=0.9, loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate overlap
    overlap = results['overlap']
    fidelity = results['fidelity']
    ax.text(0.98, 0.88,
            f'Overlap = {overlap:.4f}\nFidelity = {fidelity:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    ax.set_xlim(-0.6, n - 0.4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_target_vs_achieved.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# FIGURE 3: Fidelity Comparison (Ideal vs Quebec)
# =========================================================================

def fig3_fidelity_comparison(results):
    """Grouped bar chart showing ideal vs noisy fidelity across metrics."""
    print("  [3/3] Fidelity comparison chart...")

    noisy = results['noisy_quebec']

    # Metrics to show
    metrics = ['State fidelity\n(density matrix)', 'Hellinger fidelity\n(exact / DM)', 'Hellinger fidelity\n(8192 shots)']
    ideal_vals = [
        noisy.get('sf_target_ideal', results['fidelity']),
        noisy['hf_target_ideal'],
        noisy['hf_target_aer_shots'],
    ]
    noisy_vals = [
        noisy.get('sf_target_noisy', 0),
        noisy.get('hf_target_noisy_dm', 0),
        noisy['hf_target_noisy_shots'],
    ]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width / 2, ideal_vals, width,
                   label='Ideal (statevector)', color=C_AAE, alpha=0.88, edgecolor='none')
    bars2 = ax.bar(x + width / 2, noisy_vals, width,
                   label='FakeQuebec (noisy)', color=C_AMP, alpha=0.88, edgecolor='none')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Fidelity')
    ax.set_title('Fidelity comparison: ideal vs FakeQuebec noisy simulation',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(framealpha=0.9, fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.12)

    # Annotation explaining shot-based degradation
    ax.annotate('Shot noise\ndepresses\nfidelity',
                xy=(2, max(ideal_vals[2], noisy_vals[2]) + 0.03),
                xytext=(2.35, 0.75),
                fontsize=9, color='#888888', ha='center',
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=0.8))

    # Summary metrics
    sf_drop = ideal_vals[0] - noisy_vals[0] if noisy_vals[0] else 0
    hf_drop = ideal_vals[1] - noisy_vals[1] if noisy_vals[1] else 0
    summary = f'Noise drop (state F): {sf_drop:.3f}\nNoise drop (Hellinger DM): {hf_drop:.3f}'
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=9,
            color='#666666', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#dddddd'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_fidelity_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  GENERATING PAPER FIGURES — Section II: Quantum State Encoding")
    print("=" * 70)

    results = load_results()
    print(f"\n  Loaded results: {results['dataset']}")
    print(f"  Overlap: {results['overlap']:.6f}, Fidelity: {results['fidelity']:.6f}")

    fig1_encoding_comparison_grid(results)
    fig2_target_vs_achieved(results)
    fig3_fidelity_comparison(results)

    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
