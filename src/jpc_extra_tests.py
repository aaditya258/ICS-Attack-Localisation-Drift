"""Analyses of the manuscript that are computed from saved per-event results only (no datasets needed).

  1. Pre-registered WADI claims A and B (protocol/PREREG_WADI.md)
  2. WADI 13-attack set without attack 13 (shifted label), without attack 8 (reported unsuccessful) and without both
  3. Recursive RBC with and without the 300 s gap on the 120 injected sensor faults (paired Wilcoxon)
  4. Five-minute horizon on the events that last at least 330 s

Usage (repository root):  python src/jpc_extra_tests.py  [results_dir]
By default it reads ./reference_results. Pass ./results to check a fresh run.
"""
import os, sys
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.environ.get("LENSR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference_results")
GT = os.path.join(ROOT, "groundtruth")
LENSR, NOGAP, GAP = "LENS-R (full)", "Recursive RBC (no gap)", "Recursive RBC (300 s gap)"


def claims():
    print("=== 1. Pre-registered WADI claims")
    s = pd.read_csv(f"{R}/wadi_primary_summary.csv", index_col=0)
    t = pd.read_csv(f"{R}/wadi_primary_tests.csv")
    against = t[t.mean_diff < 0]
    sig_worse = against[against.p < 0.05]
    higher_p3 = s.loc["LENS-R T2 (full)", "MRR"] > s.loc["Simple drift fix", "MRR"]
    print(f"A: baselines significantly better than LENS-R: {len(sig_worse)}; "
          f"LENS-R {s.loc['LENS-R T2 (full)', 'MRR']:.3f} vs sliding baseline {s.loc['Simple drift fix', 'MRR']:.3f}"
          f" -> claim A {'met' if len(sig_worse) == 0 and higher_p3 else 'not met'}")
    d = pd.read_csv(f"{R}/wadi_inject_summary.csv", index_col=0)
    drop = (d["0"] - d["10"]).round(3)
    ok = drop["LENS-R T2 (full)"] < 0.05 and drop["GDN-style forecast residual"] > 0.10 and drop["LENS-R without drift correction"] > 0.10
    print("B: MRR loss from m=0 to m=10: " + ", ".join(f"{k} {v:.3f}" for k, v in drop.items()))
    print(f"   -> claim B {'met' if ok else 'not met'}")


def wadi_exclusions():
    print("\n=== 2. WADI 13-attack set with exclusions (MRR)")
    A = pd.read_csv(f"{R}/adaptive_wadi_per_attack.csv")
    A = A[(A.m == 0) & (A.set == "WADI-13 (secondary)")][["attack", "method", "rr"]]
    B = pd.read_csv(f"{R}/wadi_secondary_keepconst_per_attack.csv")[["attack", "method", "rr"]]
    D = pd.concat([A, B[~B.method.str.startswith("LENS-R")]])
    D["attack"] = D.attack.astype(str)
    out = {}
    for name, ex in [("all 13", []), ("without 13", ["13"]), ("without 8", ["8"]), ("without 8 and 13", ["8", "13"])]:
        out[name] = D[~D.attack.isin(ex)].groupby("method").rr.mean()
    T = pd.DataFrame(out).sort_values("all 13", ascending=False).round(3)
    print(T.to_string())


def gap_on_faults():
    print("\n=== 3. Recursive RBC with gap vs without gap, injected faults (pooled 120)")
    F = pd.read_csv(f"{R}/adaptive_faults_per_injection.csv")
    for sh in ["step", "ramp 10 min", "ramp 30 min", "ramp 60 min"]:
        p = F[F["shape"] == sh].pivot_table(index=["ds", "inj"], columns="method", values="rr")
        a, b = p[GAP], p[NOGAP]
        print(f"{sh:12s} gap {a.mean():.3f}  no gap {b.mean():.3f}  gap better/worse {(a > b).sum()}/{(a < b).sum()}"
              f"  p = {wilcoxon(a, b).pvalue:.4f}")


def long_events():
    print("\n=== 4. Five-minute horizon, events lasting at least 330 s")
    h = pd.read_csv(f"{R}/adaptive_horizon_per_attack.csv")
    s = pd.read_csv(f"{GT}/swat_a1_attacks.csv"); a = pd.read_csv(f"{GT}/hai2204_attacks.csv")
    dur = {f"swat_{r.attacks}": r.duration_s for r in s.itertuples()}
    dur.update({f"hai_{r.id}": r.duration_s for r in a.itertuples()})
    L = h[h.attack.map(dur) >= 330]
    g = L.groupby("method")[["h30", "last30"]].mean()
    g["change"] = g.last30 - g.h30
    print(f"events: {L.attack.nunique()}"); print(g.round(3).to_string())
    p = L.pivot(index="attack", columns="method", values="last30")
    print(f"last 30 s, {GAP} vs {LENSR}: p = {wilcoxon(p[GAP], p[LENSR]).pvalue:.3f}")


if __name__ == "__main__":
    claims(); wadi_exclusions(); gap_on_faults(); long_events()
