"""
run_full_pipeline.py — One-command reproducibility driver
=========================================================
Runs the CodonQSVT analysis end to end, in dependency order, by invoking the
individual scripts as subprocesses (clean process per stage: no shared global
state, no matplotlib backend clashes, independent failure isolation).

Order:
  1. AAE training / cache            (scripts/aae_results_gapdh.py)
  2. Pauli truncation norms          (scripts/pauli_truncation_norms.py)
  3. QSVT-vs-QSP t-sweep (+Hellinger) (scripts/tsweep_qsvt_vs_qsp_hellinger.py)
  4. Far-from-equilibrium experiment (scripts/far_from_equilibrium.py)
  5. Figures from the generated JSON (scripts/plot_hellinger_and_norm.py,
                                      scripts/paper_figures.py)

Data-generating stages run before plotting stages, because the plotters read
the JSON artifacts the sweeps write into results/.

Usage:
    python scripts/run_full_pipeline.py                # run everything
    python scripts/run_full_pipeline.py --skip figures # run all but plotting
    python scripts/run_full_pipeline.py --only sweep ffe
    python scripts/run_full_pipeline.py --keep-going    # don't stop on first failure
    python scripts/run_full_pipeline.py --list          # list stages and exit
    python scripts/run_full_pipeline.py --dry-run       # print commands only

Run from the project root so that `src` and `data` are importable.
"""

import os
import sys
import time
import argparse
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# (key, human description, script filename, optional extra args)
# Keys let you --only / --skip individual stages or whole groups.
STAGES = [
    ("aae",     "AAE training / cache (best_aae_params_gapdh_probability.json)",
        "aae_results_gapdh.py", []),
    ("pauli",   "Pauli truncation norms sweep",
        "pauli_truncation_norms.py", []),
    ("sweep",   "QSVT-vs-QSP t-sweep + Hellinger + norm decay",
        "tsweep_qsvt_vs_qsp_hellinger.py", []),
    ("ffe",     "Far-from-equilibrium trajectory (delta-start dynamics)",
        "far_from_equilibrium.py", []),
    ("table9",  "QSVT truncation-threshold validity audit",
        "threshold_sweep.py", []),
    ("plots",   "Plot Hellinger + norm-decay figures from JSON",
        "plot_hellinger_and_norm.py", []),
    ("figures", "Regenerate remaining paper figures",
        "paper_figures.py", []),
]

# Group aliases for --only / --skip convenience.
GROUPS = {
    "data":    {"aae", "pauli", "sweep", "ffe", "table9"},
    "figures": {"plots", "figures"},
    "all":     {s[0] for s in STAGES},
}


def _resolve_keys(values):
    """Expand group aliases (e.g. 'figures') into concrete stage keys."""
    resolved = set()
    valid = {s[0] for s in STAGES} | set(GROUPS)
    for v in values:
        if v not in valid:
            raise SystemExit(
                f"Unknown stage/group '{v}'. "
                f"Valid: {sorted({s[0] for s in STAGES})} "
                f"or groups {sorted(GROUPS)}.")
        if v in GROUPS:
            resolved |= GROUPS[v]
        else:
            resolved.add(v)
    return resolved


def main():
    ap = argparse.ArgumentParser(
        description="Run the full CodonQSVT pipeline in dependency order.")
    ap.add_argument("--only", nargs="+", metavar="STAGE",
                    help="run only these stages/groups")
    ap.add_argument("--skip", nargs="+", metavar="STAGE",
                    help="run everything except these stages/groups")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a stage fails (default: stop)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without executing")
    ap.add_argument("--list", action="store_true",
                    help="list the stages and exit")
    args = ap.parse_args()

    if args.list:
        print("Stages (in execution order):")
        for key, desc, script, _ in STAGES:
            print(f"  {key:9s} {script:38s} {desc}")
        print(f"\nGroups: " + ", ".join(f"{g}={sorted(s)}" for g, s in GROUPS.items()))
        return 0

    only = _resolve_keys(args.only) if args.only else None
    skip = _resolve_keys(args.skip) if args.skip else set()

    selected = []
    for stage in STAGES:
        key = stage[0]
        if only is not None and key not in only:
            continue
        if key in skip:
            continue
        selected.append(stage)

    if not selected:
        print("No stages selected. Use --list to see options.")
        return 1

    print("=" * 72)
    print("  CodonQSVT — FULL PIPELINE")
    print(f"  project root : {_PROJECT_DIR}")
    print(f"  python       : {sys.executable}")
    print(f"  stages       : {', '.join(s[0] for s in selected)}")
    print(f"  on failure   : {'continue' if args.keep_going else 'stop'}")
    print("=" * 72)

    results = []
    pipeline_t0 = time.time()

    for key, desc, script, extra in selected:
        script_path = os.path.join(_SCRIPT_DIR, script)
        if not os.path.isfile(script_path):
            print(f"\n[{key}] SKIP — script not found: {script}")
            results.append((key, "missing", 0.0))
            if not args.keep_going:
                print("  Stopping (missing script). Use --keep-going to continue.")
                break
            continue

        cmd = [sys.executable, script_path] + extra
        print(f"\n{'-'*72}\n[{key}] {desc}\n  $ {' '.join(cmd)}\n{'-'*72}")

        if args.dry_run:
            results.append((key, "dry-run", 0.0))
            continue

        t0 = time.time()
        # Run from project root so `src`/`data` imports resolve; stream output live.
        proc = subprocess.run(cmd, cwd=_PROJECT_DIR)
        dt = time.time() - t0

        if proc.returncode == 0:
            print(f"\n[{key}] OK  ({dt:.1f}s)")
            results.append((key, "ok", dt))
        else:
            print(f"\n[{key}] FAILED (exit {proc.returncode}, {dt:.1f}s)")
            results.append((key, f"fail({proc.returncode})", dt))
            if not args.keep_going:
                print("\nStopping on first failure. Re-run with --keep-going to")
                print("continue past failures, or --only to target one stage.")
                break

    total = time.time() - pipeline_t0
    print("\n" + "=" * 72)
    print("  PIPELINE SUMMARY")
    print("=" * 72)
    for key, status, dt in results:
        print(f"  {key:9s} {status:12s} {dt:7.1f}s")
    print(f"  {'-'*40}")
    print(f"  total {total:.1f}s")

    any_fail = any(st.startswith("fail") or st == "missing" for _, st, _ in results)
    if any_fail and not args.dry_run:
        print("\n  Some stages did not complete. See logs above.")
        return 1
    print("\n  Artifacts in results/ : *.json data + *.png figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
