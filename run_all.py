"""Run every experiment of the paper in order, after prepare_data.py.

Usage:
    python run_all.py              # everything (about 1 to 1.5 hours on a 2-core laptop)
    python run_all.py --from 8     # resume from step 8
    python run_all.py --only 10    # a single step
    python run_all.py --list       # show the steps

Outputs go to ./results, ./cache and ./figures. The outputs reported in the paper are in ./reference_results.
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

STEPS = [
    # (script and arguments, what it produces)
    (["prepare.py", "swat"], "SWaT: normal model, Random Forest and XGBoost detectors, TreeSHAP windows (cache)"),
    (["prepare.py", "hai"], "HAI: same as above"),
    (["evaluate.py", "10"], "Earlier design and baselines over 30 s, supervised detector results"),
    (["evaluate.py", "5"], "Horizon sensitivity for the earlier design (5 updates)"),
    (["evaluate.py", "20"], "Horizon sensitivity for the earlier design (20 updates)"),
    (["drift_run.py", "10"], "Drift-aware variants (development)"),
    (["select_variant.py"], "Variant selection on validation attacks (development)"),
    (["run2.py"], "Earlier pre-registered design (did not meet its criterion), forecast and kNN baselines"),
    (["inject.py"], "HAI drift injection for earlier designs and baselines"),
    (["run_recon.py"], "LENS-R on SWaT and HAI: accuracy, ablation, drifting tags"),
    (["inject_recon.py"], "HAI drift injection for LENS-R"),
    (["wadi_run.py"], "WADI pre-registered run with drift injection"),
    (["sweep.py"], "Sensitivity of LENS-R to its settings"),
    (["timing_online.py"], "Per-update cost of LENS-R and baselines"),
    (["diagnostics.py"], "Analyser drift share, smearing example, miss analysis"),
    (["adaptive_run.py"], "Post hoc: static and recursive RBC on all testbeds, slow drift, five-minute horizon, cost"),
    (["adaptive_faults.py"], "Post hoc: injected sensor faults and drift near attack onset"),
    (["jpc_extra_tests.py", "results"], "Pre-registered claims, WADI exclusions, gap test, long-event horizon"),
    (["jpc_figures.py", "--case"], "Figures 1 to 4 of the manuscript as PDF"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", type=int, default=1)
    p.add_argument("--only", type=int)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for i, (cmd, what) in enumerate(STEPS, 1):
            print(f"{i:2d}. {' '.join(cmd):28s} {what}")
        return
    for d in ("results", "cache", "figures"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""), LENSR_ROOT=ROOT)
    todo = [a.only] if a.only else range(a.start, len(STEPS) + 1)
    t0 = time.time()
    for i in todo:
        cmd, what = STEPS[i - 1]
        print(f"\n=== Step {i}/{len(STEPS)}: {' '.join(cmd)} -- {what}", flush=True)
        t = time.time()
        log = os.path.join(ROOT, "results", f"log_step{i:02d}_{cmd[0].replace('.py', '')}.txt")
        with open(log, "w") as fh:
            r = subprocess.run([sys.executable, os.path.join(SRC, cmd[0])] + cmd[1:], cwd=ROOT, env=env,
                               stdout=fh, stderr=subprocess.STDOUT)
        print(f"    finished in {time.time() - t:.0f} s, log: {os.path.relpath(log, ROOT)}", flush=True)
        if r.returncode != 0:
            print(f"    FAILED (exit code {r.returncode}). See the log above.")
            sys.exit(r.returncode)
    print(f"\nAll requested steps finished in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
