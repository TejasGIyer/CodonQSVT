"""
Plot Hellinger fidelity + evolved-state-norm figures
=====================================================
Reads results/tsweep_hellinger_and_norm.json and produces two PNGs that
match the target figures exactly.

Outputs:
    results/fig_hellinger_fidelity.png
    results/fig_evolved_state_norm.png

Run after `tsweep_qsvt_vs_qsp_hellinger.py`:
    python scripts/plot_hellinger_and_norm.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

JSON_PATH = os.path.join(
    _PROJECT_DIR, 'results', 'tsweep_hellinger_and_norm.json')

# colours pulled from the target plots
COLOR_QSP   = '#b3242b'   # dark red
COLOR_QSVT  = '#1f4e9d'   # dark blue
COLOR_CEIL  = '#666666'   # grey, dashed
COLOR_ENV   = '#444444'   # grey, dotted


def load_data():
    if not os.path.exists(JSON_PATH):
        sys.exit(
            f"Missing {JSON_PATH}.\n"
            f"Run  python scripts/tsweep_qsvt_vs_qsp_hellinger.py  first."
        )
    with open(JSON_PATH) as f:
        return json.load(f)


# =====================================================================
# Plot 1 — Hellinger fidelity vs evolution time
# =====================================================================
def plot_hellinger(data, out_path):
    cfg = data['config']
    rows = data['rows']

    ts     = [r['t']                    for r in rows]
    f_ceil = [r['f_classical_ceiling']  for r in rows]
    f_qsp  = [r['f_hellinger_qsp']      for r in rows]
    f_qsvt = [r['f_hellinger_qsvt']     for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ts, f_ceil, linestyle='--', marker='s',
            mfc='white', mec=COLOR_CEIL, color=COLOR_CEIL,
            markersize=6, linewidth=1.2,
            label=r'Classical ceiling $F(\pi(0), \pi_\mathrm{eq})$')

    ax.plot(ts, f_qsp, marker='o', color=COLOR_QSP,
            markersize=7, linewidth=1.8,
            label=r'QSP $e^{-iHt}$ (oscillatory, unphysical)')

    ax.plot(ts, f_qsvt, marker='^', color=COLOR_QSVT,
            markersize=7, linewidth=1.8,
            label=r'QSVT $e^{Ht}$ (stable, reweighted)')

    # Annotate the QSVT point at t = 0.5 (matches the on-figure callout)
    annot = next((r for r in rows if abs(r['t'] - 0.5) < 1e-9), None)
    if annot is not None and not np.isnan(annot['f_hellinger_qsvt']):
        f05 = annot['f_hellinger_qsvt']
        ax.annotate(
            rf"$F_H = {f05:.3f}$ at $t = 0.5$",
            xy=(0.5, f05),
            xytext=(0.55, f05 + 0.04),
            fontsize=11, color=COLOR_QSVT,
            arrowprops=dict(arrowstyle='->', color=COLOR_QSVT, lw=0.8),
        )

    title = (rf"Hellinger fidelity vs evolution time "
             rf"(threshold {cfg['threshold']:.2f}, "
             rf"{cfg['n_layers']}-layer AAE, "
             rf"$\mathcal{{O}} = {cfg['aae_overlap']:.3f}$)")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r'Evolution time $t$', fontsize=12)
    ax.set_ylabel(r'Hellinger fidelity $F_H$ vs CTMC', fontsize=12)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlim(-0.05, 2.1)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# =====================================================================
# Plot 2 — Evolved-state norm vs evolution time
# =====================================================================
def plot_norm(data, out_path):
    cfg = data['config']
    rows = data['rows']

    ts        = np.array([r['t']                              for r in rows])
    qsp_n2    = np.array([r['qsp_norm2']                      for r in rows])
    qsvt_n2   = np.array([r['qsvt_norm2']                     for r in rows])
    envelope  = np.array([r['envelope_exp_neg_2lambda_bar_t'] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ts, qsp_n2, marker='o', color=COLOR_QSP,
            markersize=7, linewidth=1.8,
            label=r'QSP $\Vert e^{-iHt}|\psi\rangle\Vert^2$ (constant $\approx 1$)')

    ax.plot(ts, qsvt_n2, marker='^', color=COLOR_QSVT,
            markersize=7, linewidth=1.8,
            label=r'QSVT $\Vert e^{Ht}|\psi\rangle\Vert^2$ (decays)')

    # Smooth envelope curve
    t_dense = np.linspace(ts.min(), ts.max(), 200)
    env_dense = np.exp(-2.0 * cfg['lambda_bar'] * t_dense)
    ax.plot(t_dense, env_dense, linestyle=':', color=COLOR_ENV,
            linewidth=1.4,
            label=r'$\sim e^{-2\bar{\lambda} t}$ (envelope)')

    ax.set_title('', fontsize=13)
    ax.set_xlabel(r'Evolution time $t$', fontsize=12)
    ax.set_ylabel(r'Evolved state norm $\Vert\tilde{\psi}(t)\Vert^2$', fontsize=12)
    ax.set_xlim(-0.05, 2.1)
    ax.set_ylim(0.0, max(1.35, float(np.nanmax(qsp_n2)) + 0.1))
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("Loading", JSON_PATH)
    data = load_data()

    results_dir = os.path.join(_PROJECT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("\nPlot 1 / 2 — Hellinger fidelity")
    plot_hellinger(
        data, os.path.join(results_dir, 'fig_hellinger_fidelity.png'))

    print("\nPlot 2 / 2 — Evolved-state norm")
    plot_norm(
        data, os.path.join(results_dir, 'fig_evolved_state_norm.png'))

    print("\nDone.")


if __name__ == "__main__":
    main()
