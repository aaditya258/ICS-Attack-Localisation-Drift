"""Comparison of LENS-R with adaptive (recursive) monitoring. Does not modify recon.py.

Usage (from the repository root):
    PYTHONPATH=src LENSR_ROOT=. python src/adaptive_run.py            # all parts
    PYTHONPATH=src LENSR_ROOT=. python src/adaptive_run.py dev        # SWaT + HAI, 30 s and 5-min horizons
    PYTHONPATH=src LENSR_ROOT=. python src/adaptive_run.py hai_drift  # HAI drift injection
    PYTHONPATH=src LENSR_ROOT=. python src/adaptive_run.py wadi       # WADI primary/secondary + drift injection
    PYTHONPATH=src LENSR_ROOT=. python src/adaptive_run.py timing     # cost of one 3-s online update

Methods
  LENS-R (full)                   ReconModel.window(X, upd, True, True)    (frozen recon.py)
  LENS-R w/o drift correction     ReconModel.window(X, upd, False, True)
  Classical RBC                   ReconModel.window(X, upd, False, False)  (static model, single-tag RBC)
  Recursive RBC (no gap)          exponentially weighted mean m and uncentred second moment S of z,
  Recursive RBC (300 s gap)       alpha = 1 - 1/1800, m0 = 0, S0 = inv(P) - 0.01 I, recursion started at most
                                  4 h before the first update of the event; C = S - m m^T, M = inv(C + 0.01 I);
                                  score_i = b_i^2 / M_ii with b = M (z_fast - m), z_fast = mean z over the last 10 s.
                                  'no gap': model updated up to the current sample. '300 s gap': up to t - 300 s.
Update times: arange(49, len(X), 3); first 10 updates after onset (30 s). Scores normalised per update, averaged.
"""
import os, sys, time, pickle, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

_ARGV, sys.argv = sys.argv[1:], sys.argv[:1]      # evaluate.py reads sys.argv[1] at import
import data as Dm
from paths import ROOT, CACHE, RESULTS
from lens import Config, NormalModel, EPS
from recon import ReconModel
from evaluate import rank_eval, paired, boot_ci

assert hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon.py"), "rb").read()).hexdigest() == \
    "6a8fb2546c6e97d4d90ffe71d17937ced408606f96bdc73331cc2a21b2b12c19", "recon.py changed since freezing"

N = 10                       # 30 s horizon
N_LONG = 100                 # 5-min horizon
ALPHA = 1 - 1 / 1800
RIDGE = 0.01
HIST = 4 * 3600              # recursion starts at most 4 h before the first update
GAP = 300
HAI_DISTRACT = ["P1_FT02", "P1_FT03Z", "P2_24Vdc", "P2_VIBTR03", "P4_HT_PO"]
FULL, NODC, CLAS = "LENS-R (full)", "LENS-R w/o drift correction", "Classical RBC"
REC0, REC300 = "Recursive RBC (no gap)", "Recursive RBC (300 s gap)"
METHODS = [FULL, NODC, CLAS, REC0, REC300]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
os.makedirs(RESULTS, exist_ok=True)


def norm_mean(A):
    return (A / (A.sum(1, keepdims=True) + EPS)).mean(0)


def normal_model(name, ds):
    p = os.path.join(CACHE, f"{name}_models.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))["nm"]
    return NormalModel().fit(ds.Xn, Config())


# ---------------------------------------------------------------- recursive RBC
class RecursiveRBC:
    def __init__(self, nm, alpha=ALPHA, ridge=RIDGE):
        self.nm, self.alpha, self.ridge = nm, alpha, ridge
        n = len(nm.mu)
        self.S0 = np.linalg.inv(nm.P) - ridge * np.eye(n)        # uncentred second moment of normal z
        self.m0 = np.zeros(n)

    def states(self, Z, start, times):
        """EW mean and second moment after including sample t (inclusive), for each t in times.
        Recursion runs from `start`; a time before `start` returns the initial (normal) state."""
        a = self.alpha
        m, S = self.m0.copy(), self.S0.copy()
        out, k = {}, start                                       # k = next sample to include
        for t in sorted(set(int(x) for x in times)):
            if t < start:
                out[t] = (self.m0, self.S0); continue
            if t + 1 > k:                                        # include samples k .. t
                B = Z[k:t + 1]; L = len(B)
                w = (1 - a) * a ** np.arange(L - 1, -1, -1)
                m = a ** L * m + w @ B
                S = a ** L * S + (B * w[:, None]).T @ B
                k = t + 1
            out[t] = (m.copy(), S.copy())
        return out

    @staticmethod
    def score(m, S, zf, ridge=RIDGE):
        C = S - np.outer(m, m)
        M = np.linalg.inv(C + ridge * np.eye(len(m)))
        b = M @ (zf - m)
        return b ** 2 / np.diag(M)

    def window(self, X, upd, start):
        """Score matrices (len(upd), N) for the no-gap and 300 s-gap variants."""
        nm = self.nm
        lo = max(0, start)
        hi = int(upd.max()) + 1
        Z = ((X[lo:hi].astype(np.float64) - nm.mu) / nm.sd)
        u = upd - lo
        st = self.states(Z, max(0, start - lo), np.r_[u, u - GAP])
        s0, s300 = [], []
        for t in u:
            zf = Z[max(0, t - 9):t + 1].mean(0)
            s0.append(self.score(*st[t], zf))
            s300.append(self.score(*st[t - GAP], zf))
        return np.array(s0), np.array(s300)


def score_event(X, upd_all, onset, rm, rr, n=N, long=False):
    """All five methods over the first n updates after onset (plus 5-min horizon for three of them)."""
    u0 = int(np.searchsorted(upd_all, onset))
    u = upd_all[u0:u0 + n]
    out = {FULL: norm_mean(rm.window(X, u, True, True)[0]),
           NODC: norm_mean(rm.window(X, u, False, True)[0]),
           CLAS: norm_mean(rm.window(X, u, False, False)[0])}
    ul = upd_all[u0:u0 + (N_LONG if long else n)]
    S0, S300 = rr.window(X, ul, int(ul[0]) - HIST)
    out[REC0], out[REC300] = norm_mean(S0[:n]), norm_mean(S300[:n])
    if long:
        SL = rm.window(X, ul, True, True)[0]
        lg = {FULL: SL, REC0: S0, REC300: S300}
        out["_5min"] = {k: norm_mean(v) for k, v in lg.items()}
        out["_last30"] = {k: norm_mean(v[-N:]) for k, v in lg.items()}       # updates 91-100 only
    return out


def table(per, label):
    g = per.groupby("method")
    t = pd.DataFrame({"n": g.size(), "MRR": g.rr.mean(), "Top@1": g.top1.mean(), "Top@3": g.top3.mean(),
                      "Top@5": g.top5.mean()})
    ci = {k: boot_ci(v.rr.values) for k, v in g}
    t["CI low"] = [ci[m][0] for m in t.index]; t["CI high"] = [ci[m][1] for m in t.index]
    t = t.reindex([m for m in METHODS if m in t.index])
    t.insert(0, "set", label)
    return t


def tests(per, label, a=FULL):
    r = pd.DataFrame([paired(per, a, b) for b in METHODS if b != a and b in set(per.method)])
    r.insert(0, "set", label)
    return r


# ---------------------------------------------------------------- parts
def part_dev():
    rows, hz = [], []
    for name in os.environ.get("LENSR_DEV_SETS", "swat,hai").split(","):
        ds = Dm.load(name); nm = normal_model(name, ds)
        rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds); rr = RecursiveRBC(nm)
        att = ds.attacks[ds.attacks.localisable].reset_index(drop=True)
        log(name, "models fitted;", len(att), "events")
        for _, a in att.iterrows():
            X = ds.streams[a.stream].X
            upd_all = np.arange(49, len(X), 3)
            sc = score_event(X, upd_all, a.onset, rm, rr, long=True)
            ev = lambda s: rank_eval(s, ds.tags, ds.stage, a.points, a.stages)
            aid = f"{name}_{a.label}"
            for m in METHODS:
                rows.append(dict(ds=name, attack=aid, method=m, **ev(sc[m])))
            for m in [FULL, REC0, REC300]:
                hz.append(dict(ds=name, attack=aid, method=m, h30=ev(sc[m])["rr"],
                               h5min=ev(sc["_5min"][m])["rr"], last30=ev(sc["_last30"][m])["rr"]))
        log(name, "done")
    per = pd.DataFrame(rows); per.to_csv(f"{RESULTS}/adaptive_dev_per_attack.csv", index=False)
    T = pd.concat([table(per[per.ds == "swat"], "SWaT (33)"), table(per[per.ds == "hai"], "HAI (58)"),
                   table(per, "Pooled (91)")])
    P = pd.concat([tests(per[per.ds == "swat"], "SWaT (33)"), tests(per[per.ds == "hai"], "HAI (58)"),
                   tests(per, "Pooled (91)")])
    T.round(3).to_csv(f"{RESULTS}/adaptive_dev_summary.csv"); P.to_csv(f"{RESULTS}/adaptive_dev_tests.csv", index=False)
    print("\n=== Accuracy, 30 s (first 10 updates) ===\n", T.round(3).to_string())
    print("\n=== Wilcoxon, LENS-R (full) vs each ===\n", P[["set", "b", "n", "mean_diff", "wins", "losses", "ties", "p"]].to_string())
    H = pd.DataFrame(hz); H.to_csv(f"{RESULTS}/adaptive_horizon_per_attack.csv", index=False)
    h = H.groupby("method")[["h30", "h5min", "last30"]].mean().reindex([FULL, REC0, REC300])
    h["5min - 30s"] = h.h5min - h.h30; h["last30 - 30s"] = h.last30 - h.h30
    def hp(m, col):
        d = H[H.method == m]
        long_ = pd.concat([d.assign(method="h30", rr=d.h30), d.assign(method=col, rr=d[col])])
        return paired(long_, "h30", col)["p"]
    h["p (30s vs 5min)"] = pd.Series({m: hp(m, "h5min") for m in [FULL, REC0, REC300]})
    h["p (30s vs last30)"] = pd.Series({m: hp(m, "last30") for m in [FULL, REC0, REC300]})
    h.round(4).to_csv(f"{RESULTS}/adaptive_horizon.csv")
    print("\n=== Horizon, pooled 91: MRR at 30 s, over 5 min, and over updates 91-100 only ===\n", h.round(3).to_string())
    # sanity check against the published per-attack results
    ref = os.path.join(ROOT, "reference_results", "recon_per_attack.csv")
    if os.path.exists(ref):
        R = pd.read_csv(ref)
        for mine, theirs in [(FULL, "LENS-R T2 (full)"), (NODC, "LENS-R T2 (no drift correction)"), (CLAS, "LENS-R T2 (classic single-tag RBC)")]:
            a = per[per.method == mine].set_index("attack").rr
            b = R[R.method == theirs].set_index("attack").rr
            j = a.index.intersection(b.index)
            print(f"sanity {mine:28s} events matched {len(j):3d}  max |rr diff| {np.abs(a[j] - b[j]).max():.6f}  "
                  f"MRR here {a[j].mean():.3f}  reference {b[j].mean():.3f}")
    return per


def part_hai_drift():
    ds = Dm.load("hai"); nm = normal_model("hai", ds)
    rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds); rr = RecursiveRBC(nm)
    cols = [ds.tags.index(t) for t in HAI_DISTRACT]
    rows = []
    for m in [0, 3, 6, 10]:
        for sname, s in ds.streams.items():
            X = s.X.copy()
            if m:
                X[:, cols] += (m * nm.sd[cols])[None, :].astype(np.float32) * np.minimum(1, np.arange(len(X)) / 7200)[:, None]
            upd_all = np.arange(49, len(X), 3)
            for _, a in ds.attacks[ds.attacks.stream == sname].iterrows():
                sc = score_event(X, upd_all, a.onset, rm, rr)
                for k in METHODS:
                    rows.append(dict(m=m, method=k, attack=a.label, **rank_eval(sc[k], ds.tags, ds.stage, a.points, a.stages)))
        log("HAI drift m =", m, "done")
    df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/adaptive_hai_drift_per_attack.csv", index=False)
    t = drift_table(df); t.to_csv(f"{RESULTS}/adaptive_hai_drift.csv")
    print("\n=== HAI drift injection (58 events): MRR vs drift size ===\n", t.to_string())


def drift_table(df):
    t = df.groupby(["method", "m"]).rr.mean().unstack("m").reindex(METHODS)
    t.columns = [f"{c} sigma" for c in t.columns]
    t["drop 0->10"] = t.iloc[:, -1] - t.iloc[:, 0]
    p = {k: paired(df[df.m.isin([0, 10])].query("method == @k").assign(method=lambda d: d.m.astype(str)), "0", "10")["p"]
         for k in METHODS}
    t["p (0 vs 10)"] = pd.Series(p)
    return t.round(3)


def part_wadi():
    import wadi_run as W
    out = []
    for keep in [False, True]:
        tag = "WADI-13 (secondary)" if keep else "WADI-8 (primary)"
        D = W.load(keep); tags = D["tags"]; stage = np.array([t.split("_")[0] for t in tags])
        att = W.attacks(D)
        att = att[att.points.apply(lambda ps: any(p in tags for p in ps))].reset_index(drop=True)
        nm = NormalModel().fit(D["Xn"], Config())
        rm = ReconModel().fit(D["Xn"], nm, "T2", D["bounds"]); rr = RecursiveRBC(nm)
        Xa = D["Xa"]; upd_all = np.arange(49, len(Xa), 3)
        log(tag, "events", len(att), "tags", len(tags))
        levels = [0, 6, 10] if not keep else [0]
        distract = None
        if not keep:
            targets = {p for ps in att.points for p in ps}
            nuniq = np.array([len(np.unique(D["Xn"][::50, j])) for j in range(len(tags))])
            cont = [j for j, t in enumerate(tags) if nuniq[j] > 6 and t not in targets]
            distract = sorted(np.random.default_rng(0).choice(cont, 5, replace=False).tolist())
            log("distractors", [tags[j] for j in distract])
        for m in levels:
            X = Xa.copy()
            if m:
                X[:, distract] += (m * nm.sd[distract])[None, :].astype(np.float32) * np.minimum(1, np.arange(len(X)) / 7200)[:, None]
            for _, a in att.iterrows():
                sc = score_event(X, upd_all, a.onset, rm, rr)
                pts = [p for p in a.points if p in tags]
                for k in METHODS:
                    out.append(dict(set=tag, m=m, attack=str(a.attack), method=k, **rank_eval(sc[k], tags, stage, pts, a.stages)))
            log(tag, "m =", m, "done")
    df = pd.DataFrame(out); df.to_csv(f"{RESULTS}/adaptive_wadi_per_attack.csv", index=False)
    T = pd.concat([table(df[(df.set == s) & (df.m == 0)], s) for s in ["WADI-8 (primary)", "WADI-13 (secondary)"]])
    P = pd.concat([tests(df[(df.set == s) & (df.m == 0)], s) for s in ["WADI-8 (primary)", "WADI-13 (secondary)"]])
    T.round(3).to_csv(f"{RESULTS}/adaptive_wadi_summary.csv"); P.to_csv(f"{RESULTS}/adaptive_wadi_tests.csv", index=False)
    print("\n=== WADI accuracy, 30 s ===\n", T.round(3).to_string())
    print(P[["set", "b", "n", "mean_diff", "wins", "losses", "ties", "p"]].to_string())
    d = drift_table(df[df.set == "WADI-8 (primary)"]); d.to_csv(f"{RESULTS}/adaptive_wadi_drift.csv")
    print("\n=== WADI-8 drift injection: MRR vs drift size ===\n", d.to_string())


def part_timing():
    from recon import greedy, _cum, _wmean, Q_MAX, K_FAST
    rows = []
    with threadpool_limits(1):
        for name in os.environ.get("LENSR_DEV_SETS", "swat,hai").split(","):
            ds = Dm.load(name); nm = normal_model(name, ds)
            rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds); rr = RecursiveRBC(nm)
            X = list(ds.streams.values())[-1].X
            Z = (X.astype(np.float64) - nm.mu) / nm.sd
            n = Z.shape[1]
            # recursive RBC: 3 rank-one updates of m and S, one inverse, scoring
            m, S = rr.m0.copy(), rr.S0.copy(); a = ALPHA
            st = {"k": 4000, "m": m, "S": S}

            def rec_update():
                k = st["k"]
                for j in range(3):
                    z = Z[k + j]
                    st["m"] = a * st["m"] + (1 - a) * z
                    st["S"] = a * st["S"] + (1 - a) * np.outer(z, z)
                st["k"] = k + 3
                zf = Z[k - 7:k + 3].mean(0)
                rr.score(st["m"], st["S"], zf)
            C = _cum(Z)
            us = np.random.default_rng(0).integers(4000, len(X) - 10, 50)
            zf = _wmean(C, us + 1 - 10, us + 1); zs = _wmean(C, us - 1800, us - 300)
            ii = [0]

            def lens_full():
                i = ii[0] % len(us); ii[0] += 1
                z = zf[i].copy(); _, Q, fQ, _ = greedy(zs[i], rm.M, Q_MAX, thr=rm.thr_slow)
                if len(Q): z[Q] -= fQ
                greedy(z, rm.M, K_FAST)

            def classical():
                i = ii[0] % len(us); ii[0] += 1; greedy(zf[i], rm.M, 0)
            for meth, f, reps in [(REC0, rec_update, 1000), (FULL + " (same machine)", lens_full, 50),
                                  (CLAS, classical, 200)]:
                f(); t = time.perf_counter()
                for _ in range(reps): f()
                rows.append(dict(ds=name, tags=n, method=meth, ms=(time.perf_counter() - t) / reps * 1000))
            log(name, "timed")
    df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/adaptive_timing.csv", index=False)
    import platform
    print("\n=== Time per 3-s update (ms, single thread) ===\n",
          df.pivot(index="method", columns="ds", values="ms").round(3).to_string(),
          f"\n{platform.machine()} {os.cpu_count()} cores; the recursive-RBC cost is the same for both gap variants")


if __name__ == "__main__":
    parts = _ARGV or ["dev", "hai_drift", "wadi", "timing"]
    for p in parts:
        {"dev": part_dev, "hai_drift": part_hai_drift, "wadi": part_wadi, "timing": part_timing}[p]()
