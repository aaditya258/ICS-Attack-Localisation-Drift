"""Follow-up to adaptive_run.py: where should LENS-R differ from adaptive (recursive) monitoring?

  faults : synthetic step and incipient (ramp) sensor faults of 4 sigma on one known tag, injected into
           attack-free stretches of the SWaT, HAI and WADI test streams (40 injections per plant, seed 0).
           Ramp durations 10, 30, 60 min. Each method is scored over the 10 updates (30 s) that start when the
           fault reaches its full size. The same (location, tag) pairs are used for every fault shape.
  drift  : drift on the same five non-target distractor tags as before (HAI list, WADI seed-0 choice), but now
           close to each attack onset: a step that occurred 2, 10, 20 or 60 min before onset, or a 20-min ramp
           ending at onset; size 6 and 10 sigma. Scored on the first 10 updates after onset.

Usage (repository root):  PYTHONPATH=src LENSR_ROOT=. python src/adaptive_faults.py [faults] [drift]
recon.py is not modified.
"""
import sys
_ARGV, sys.argv = sys.argv[1:], sys.argv[:1]
import numpy as np
import pandas as pd
import data as Dm
import wadi_run as W
from lens import Config, NormalModel
from recon import ReconModel
from evaluate import rank_eval, paired, boot_ci
from adaptive_run import (RecursiveRBC, score_event, normal_model, METHODS, FULL, HIST, N, HAI_DISTRACT,
                          RESULTS, log)

K_INJ, SIZE, SEED = 40, 4.0, 0
SHAPES = {"step": 0, "ramp 10 min": 600, "ramp 30 min": 1800, "ramp 60 min": 3600}
CLEAN_BEFORE = 2 * 3600


def scored_slice(X, upd_all, onset, rm, rr, inject):
    """Score one event on a slice of the stream (identical to scoring the full stream, much cheaper).
    inject(Xs, offset) modifies the slice in place; offset = index of the slice start in the stream."""
    u0 = int(np.searchsorted(upd_all, onset))
    first, last = int(upd_all[u0]), int(upd_all[min(u0 + N, len(upd_all)) - 1])
    s = max(0, first - HIST - 60)
    Xs = X[s:last + 1].copy()
    inject(Xs, s)
    ua = upd_all[(upd_all >= s) & (upd_all <= last)] - s
    return score_event(Xs, ua, onset - s, rm, rr)


def plants():
    """(name, tags, stage, nm, rm, rr, streams [(X, y)], attacks per stream, continuous tags)"""
    out = []
    for name in ["swat", "hai"]:
        ds = Dm.load(name); nm = normal_model(name, ds)
        rm = ReconModel().fit(ds.Xn, nm, "T2", ds.normal_bounds)
        nuniq = np.array([len(np.unique(ds.Xn[::50, j])) for j in range(len(ds.tags))])
        out.append(dict(name=name, tags=ds.tags, stage=ds.stage, nm=nm, rm=rm, rr=RecursiveRBC(nm),
                        streams={k: (s.X, s.y) for k, s in ds.streams.items()}, attacks=ds.attacks, ds=ds,
                        cont=[j for j in range(len(ds.tags)) if nuniq[j] > 6]))
        log(name, "loaded")
    D = W.load(False); tags = D["tags"]
    nm = NormalModel().fit(D["Xn"], Config())
    nuniq = np.array([len(np.unique(D["Xn"][::50, j])) for j in range(len(tags))])
    out.append(dict(name="wadi", tags=tags, stage=np.array([t.split("_")[0] for t in tags]), nm=nm,
                    rm=ReconModel().fit(D["Xn"], nm, "T2", D["bounds"]), rr=RecursiveRBC(nm),
                    streams={"attack": (D["Xa"], D["y"])}, D=D, cont=[j for j in range(len(tags)) if nuniq[j] > 6]))
    log("wadi loaded")
    return out


def part_faults(P):
    rows = []
    span = max(SHAPES.values()) + 3 * N + 60
    for p in P:
        rng = np.random.default_rng(SEED)
        cand = []
        for sname, (X, y) in p["streams"].items():
            c = np.cumsum(np.r_[0, y.astype(int)])
            t = np.arange(CLEAN_BEFORE, len(y) - span)
            ok = t[(c[t + span] - c[t - CLEAN_BEFORE]) == 0]          # no attack label in [t0 - 2 h, t0 + span]
            cand += [(sname, int(v)) for v in ok[::60]]
        pick = rng.choice(len(cand), K_INJ, replace=False)
        inj = [(cand[i][0], cand[i][1], int(rng.choice(p["cont"]))) for i in sorted(pick)]
        log(p["name"], "attack-free start positions:", len(cand), "| injections:", len(inj))
        for k, (sname, t0, j) in enumerate(inj):
            X = p["streams"][sname][0]
            upd_all = np.arange(49, len(X), 3)
            for shape, T in SHAPES.items():
                def inject(Xs, off, t0=t0, j=j, T=T):
                    tt = np.arange(len(Xs)) + off - t0
                    ramp = np.clip(tt / T, 0, 1) if T else (tt >= 0).astype(float)
                    Xs[:, j] += (SIZE * p["nm"].sd[j] * ramp).astype(np.float32)
                sc = scored_slice(X, upd_all, t0 + T, p["rm"], p["rr"], inject)
                for m in METHODS:
                    rows.append(dict(ds=p["name"], shape=shape, inj=f"{p['name']}_{k}", tag=p["tags"][j], method=m,
                                     **rank_eval(sc[m], p["tags"], p["stage"], [p["tags"][j]], [p["stage"][j]])))
        log(p["name"], "faults done")
    df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/adaptive_faults_per_injection.csv", index=False)
    out = []
    for sel, label in [(df.ds == "swat", "SWaT"), (df.ds == "hai", "HAI"), (df.ds == "wadi", "WADI"), (df.ds.notna(), "Pooled")]:
        for shape in SHAPES:
            d = df[sel & (df["shape"] == shape)]
            for m in METHODS:
                r = d[d.method == m].rr
                ci = boot_ci(r.values)
                pv = paired(d.rename(columns={"inj": "attack"}), FULL, m)["p"] if m != FULL else np.nan
                out.append(dict(set=label, shape=shape, method=m, n=len(r), MRR=r.mean(), Top1=d[d.method == m].top1.mean(),
                                Top3=d[d.method == m].top3.mean(), ci_low=ci[0], ci_high=ci[1], p_vs_LENSR=pv))
    T = pd.DataFrame(out); T.round(3).to_csv(f"{RESULTS}/adaptive_faults_summary.csv", index=False)
    piv = T.pivot_table(index=["set", "method"], columns="shape", values="MRR", sort=False)[list(SHAPES)]
    print("\n=== Injected sensor faults (4 sigma, one tag): MRR ===\n", piv.round(3).to_string())
    pp = T[T.set == "Pooled"].pivot_table(index="method", columns="shape", values="p_vs_LENSR", sort=False)[list(SHAPES)]
    print("\nPooled: Wilcoxon p, LENS-R (full) vs each\n", pp.round(3).to_string())


def drift_events(p):
    """(stream X, upd_all, onset, points, stages) per scored attack, and the distractor columns."""
    if p["name"] == "hai":
        ds = p["ds"]; cols = [p["tags"].index(t) for t in HAI_DISTRACT]
        ev = [(ds.streams[a.stream].X, a.onset, a.points, a.stages) for _, a in ds.attacks.iterrows()]
    else:
        D = p["D"]; tags = p["tags"]
        att = W.attacks(D); att = att[att.points.apply(lambda ps: any(q in tags for q in ps))].reset_index(drop=True)
        targets = {q for ps in att.points for q in ps}
        nuniq = np.array([len(np.unique(D["Xn"][::50, j])) for j in range(len(tags))])
        cont = [j for j, t in enumerate(tags) if nuniq[j] > 6 and t not in targets]
        cols = sorted(np.random.default_rng(0).choice(cont, 5, replace=False).tolist())
        ev = [(D["Xa"], a.onset, [q for q in a.points if q in tags], a.stages) for _, a in att.iterrows()]
    return ev, cols


def part_drift(P):
    scen = [("none", None, 0)] + [(f"step {d} min before", "step", d * 60) for d in (2, 10, 20, 60)] + \
           [("ramp 20 min, ends at onset", "ramp", 1200)]
    rows = []
    for p in [q for q in P if q["name"] in ("hai", "wadi")]:
        ev, cols = drift_events(p)
        log(p["name"], "drift distractors:", [p["tags"][c] for c in cols], "| events:", len(ev))
        sd = p["nm"].sd[cols]
        for label, kind, d in scen:
            for m in ([0] if kind is None else [6, 10]):
                for k, (X, onset, pts, stg) in enumerate(ev):
                    upd_all = np.arange(49, len(X), 3)

                    def inject(Xs, off, onset=onset, kind=kind, d=d, m=m):
                        if kind is None:
                            return
                        tt = np.arange(len(Xs)) + off - onset          # seconds relative to attack onset
                        g = (tt >= -d).astype(float) if kind == "step" else np.clip((tt + d) / d, 0, 1)
                        Xs[:, cols] += (m * sd)[None, :].astype(np.float32) * g[:, None].astype(np.float32)
                    sc = scored_slice(X, upd_all, onset, p["rm"], p["rr"], inject)
                    for meth in METHODS:
                        rows.append(dict(ds=p["name"], scenario=label, m=m, attack=f"{p['name']}_{k}", method=meth,
                                         **rank_eval(sc[meth], p["tags"], p["stage"], pts, stg)))
                log(p["name"], label, m, "done")
    df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/adaptive_drift_speed_per_attack.csv", index=False)
    out = []
    for (ds, sc_, m), d in df.groupby(["ds", "scenario", "m"], sort=False):
        base = df[(df.ds == ds) & (df.scenario == "none")]
        for meth in METHODS:
            x = d[d.method == meth]
            b = base[base.method == meth]
            pv = paired(pd.concat([b.assign(method="base"), x.assign(method="drift")]), "base", "drift")["p"] if sc_ != "none" else np.nan
            out.append(dict(ds=ds, scenario=sc_, m=m, method=meth, MRR=x.rr.mean(),
                            change_vs_none=x.rr.mean() - b.rr.mean(), p_vs_none=pv))
    T = pd.DataFrame(out); T.round(3).to_csv(f"{RESULTS}/adaptive_drift_speed_summary.csv", index=False)
    for ds in T.ds.unique():
        piv = T[T.ds == ds].pivot_table(index=["scenario", "m"], columns="method", values="MRR", sort=False)[METHODS]
        print(f"\n=== {ds.upper()}: MRR with drift near onset (5 distractor tags) ===\n", piv.round(3).to_string())


if __name__ == "__main__":
    parts = _ARGV or ["faults", "drift"]
    P = plants()
    for q in parts:
        {"faults": part_faults, "drift": part_drift}[q](P)
