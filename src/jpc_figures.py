"""Figures of the manuscript, as vector PDF.

Usage:
    python src/jpc_figures.py              # Figs. 1, 2 and 3 (no datasets needed)
    python src/jpc_figures.py --case       # also Fig. 4 (needs prepare_data.py and prepare.py swat)

Fig. 1 (method) is black and white. Figs. 2 and 3 are drawn from ./reference_results.
Fig. 4 (case study) recomputes LENS-R on SWaT attack 8 and needs the prepared SWaT data and cache.
Output goes to ./figures.
"""
import os, sys, pickle, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
_CASE = "--case" in sys.argv
sys.argv = sys.argv[:1]                       # evaluate.py reads sys.argv[1] at import
from paths import ROOT, CACHE, FIGURES as OUT  # noqa: E402
from evaluate import boot_ci                   # noqa: E402

REF = os.path.join(ROOT, "reference_results")
os.makedirs(OUT, exist_ok=True)
USETEX = shutil.which("latex") is not None
if USETEX:
    mpl.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage[T1]{fontenc}"})
else:
    mpl.rcParams.update({"font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix"})
mpl.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402

PCT = r"\%" if USETEX else "%"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#0b0b0b"
ST = {  # label -> (colour, linestyle, marker, marker face)
    "LENS-R": (BLUE, "-", "o", BLUE),
    "LENS-R w/o drift correction": (ORANGE, "--", "o", "white"),
    "Static RBC": (GREEN, ":", "s", "white"),
    "Recursive RBC, no gap": (MAGENTA, "-.", "^", "white"),
    "Recursive RBC, 300 s gap": (AQUA, "-", "^", AQUA),
    "GDN-style residual": (YELLOW, "--", "D", YELLOW),
}
NAME = {"LENS-R (full)": "LENS-R", "LENS-R w/o drift correction": "LENS-R w/o drift correction",
        "Classical RBC": "Static RBC", "Recursive RBC (no gap)": "Recursive RBC, no gap",
        "Recursive RBC (300 s gap)": "Recursive RBC, 300 s gap",
        "Linear forecast residual (GDN-style)": "GDN-style residual",
        "GDN-style forecast residual": "GDN-style residual"}
ORDER = list(ST)


def it(s):
    return rf"\textit{{{s}}}" if USETEX else s


def bf(s):
    return rf"\textbf{{{s}}}" if USETEX else s


def fig_method():
    """Fig. 1: the localisation layer beside the detector (a) and the data each reference uses (b). Black and white."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_axes([0.0, 0.47, 1.0, 0.53]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, txt, ls="-", lw=0.7, fs=7.5, fill="white"):
        ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec="black", lw=lw, ls=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(p, q, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.7, color="black", ls=ls,
                                     shrinkA=0, shrinkB=0))

    ax.text(0.5, 29.5, "(a)", fontsize=8, va="top")
    box(4, 22, 15, 5, "Plant: sensors,\nactuators, PLCs")
    box(25, 22, 15, 5, "Historian\n(1 Hz readings)")
    box(47, 22, 17, 5, "Anomaly detector\n(unchanged)")
    arrow((19, 24.5), (25, 24.5)); arrow((40, 24.5), (47, 24.5))
    ax.add_patch(Rectangle((4, 1.5), 78, 17.5, fc="none", ec="black", lw=1.0))
    ax.text(81.3, 18.3, bf("Localisation layer"), ha="right", va="top", fontsize=8.5)
    ax.plot([32.5, 32.5], [22, 19], color="black", lw=0.7)
    ax.text(33.5, 20.5, "all readings", fontsize=7, va="center")
    arrow((32.5, 19.0), (32.5, 15.5))
    box(7, 3.5, 32, 12, "")
    ax.text(23, 13.6, "Reference of normal behaviour", ha="center", va="center", fontsize=7.5)
    items = [("\u2013 static model, fitted once", 9.5), ("\u2013 offsets adapted (LENS-R)", 9.5),
             ("\u2013 mean and covariance adapted,", 9.5), ("with or without a 5-min gap", 11.6)]
    for i, (line, x) in enumerate(items):
        ax.text(x, 11.2 - 2.0 * i, line, ha="left", va="center", fontsize=7.2)
    box(45, 6, 16, 7, "Reconstruct each\nsensor from the\nothers and rank")
    box(66, 6, 13, 7, "Ranked sensors\nwith the value\neach should read")
    arrow((39, 9.5), (45, 9.5)); arrow((61, 9.5), (66, 9.5))
    box(86, 6, 13, 7, "Operator")
    arrow((79, 9.5), (86, 9.5))
    ax.plot([64, 92.5], [24.5, 24.5], color="black", lw=0.7, ls=":")
    arrow((92.5, 24.5), (92.5, 13), ls=":")
    ax.text(78, 25.0, it("alarm"), fontsize=7, ha="center", va="bottom")

    bx = fig.add_axes([0.20, 0.07, 0.62, 0.36])
    t = np.linspace(-60, 0, 1201)
    rows = [("Static RBC", "nothing"), ("LENS-R", "offsets only"),
            ("Recursive RBC, no gap", "mean and covariance"), ("Recursive RBC, 5-min gap", "mean and covariance")]
    H = 0.62
    for k, (name, adapts) in enumerate(rows):
        y0 = 3 - k
        bx.plot([-60, 0], [y0, y0], color="0.6", lw=0.5)
        if k == 0:
            bx.text(-58.5, y0 + 0.30, "normal data only, recorded before deployment", fontsize=6.5, va="center")
        elif k == 1:
            bx.add_patch(Rectangle((-30, y0), 25, H * 0.75, fc="0.75", ec="black", lw=0.6))
        else:
            g = 5 if k == 3 else 0
            w = np.where(t <= -g, np.exp((t + g) / 30.0), 0.0)
            bx.fill_between(t, y0, y0 + H * w, color="0.75", lw=0)
            bx.plot(t, y0 + H * w, color="black", lw=0.7)
        bx.add_patch(Rectangle((-10 / 60, y0), 10 / 60 + 0.25, H, fc="black", ec="black", lw=0))
        bx.text(-61.5, y0 + 0.25, name, ha="right", va="center", fontsize=7)
        bx.text(1.5, y0 + 0.25, adapts, ha="left", va="center", fontsize=7)
    bx.axvspan(-5, 0, color="0.92", lw=0, zorder=0)
    bx.text(-2.5, 3.85, "5 min", ha="center", fontsize=6.5)
    bx.set_xlim(-60, 0.8); bx.set_ylim(-0.25, 4.05)
    bx.set_yticks([]); bx.spines["left"].set_visible(False)
    bx.set_xticks([-60, -45, -30, -15, -5, 0])
    bx.set_xlabel("Time before the current update (min). Black bar: the last 10 s that are ranked")
    bx.text(-73, 4.05, "(b)", fontsize=8, va="top")
    bx.text(1.5, 3.95, it("adapts"), fontsize=7, ha="left", va="top")
    fig.savefig(f"{OUT}/fig_method.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def fig_accuracy():
    """Fig. 2: MRR on real attacks with bootstrap 95% intervals."""
    A = pd.read_csv(f"{REF}/adaptive_dev_per_attack.csv")
    G = pd.read_csv(f"{REF}/recon_per_attack.csv")
    G = G[G.method == "Linear forecast residual (GDN-style)"]
    D = pd.concat([A[["ds", "attack", "method", "rr"]], G[["ds", "attack", "method", "rr"]]])
    W = pd.read_csv(f"{REF}/adaptive_wadi_per_attack.csv"); W = W[W.m == 0]
    G8 = pd.read_csv(f"{REF}/wadi_primary_per_attack.csv"); G13 = pd.read_csv(f"{REF}/wadi_secondary_keepconst_per_attack.csv")
    sets = {"SWaT (33)": D[D.ds == "swat"], "HAI (58)": D[D.ds == "hai"],
            "WADI (8)": pd.concat([W[W.set == "WADI-8 (primary)"], G8[G8.method == "GDN-style forecast residual"]]),
            "WADI (13)": pd.concat([W[W.set == "WADI-13 (secondary)"], G13[G13.method == "GDN-style forecast residual"]])}
    fig, ax = plt.subplots(figsize=(5.3, 2.9))
    w = 0.13
    for k, lab in enumerate(ORDER):
        xs, ys, lo, hi = [], [], [], []
        for i, (s, d) in enumerate(sets.items()):
            r = d[d.method.map(NAME) == lab].rr.values
            if not len(r):
                continue
            ci = boot_ci(r)
            xs.append(i + (k - 2.5) * w); ys.append(r.mean()); lo.append(r.mean() - ci[0]); hi.append(ci[1] - r.mean())
        ax.bar(xs, ys, w, color=ST[lab][0], edgecolor="white", linewidth=0.4, label=lab)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels(list(sets)); ax.set_ylabel(f"MRR (95{PCT} CI)")
    ax.set_ylim(0, 0.8)
    fig.legend(*ax.get_legend_handles_labels(), ncol=3, fontsize=7, loc="upper center", handlelength=1.2,
               columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(f"{OUT}/fig_accuracy.pdf"); plt.close(fig)


def fig_tradeoff():
    """Fig. 3: injected faults by growth time, and HAI attacks by age of a 10-sigma drift step."""
    F = pd.read_csv(f"{REF}/adaptive_faults_summary.csv"); F = F[F.set == "Pooled"]
    S = pd.read_csv(f"{REF}/adaptive_drift_speed_summary.csv")
    S = S[(S.ds == "hai") & (S.m.isin([0, 10])) & (~S.scenario.str.startswith("ramp"))]
    shapes = ["step", "ramp 10 min", "ramp 30 min", "ramp 60 min"]
    scen = ["step 2 min before", "step 10 min before", "step 20 min before", "step 60 min before"]
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 3.0))
    for lab in ORDER[:5]:
        c, ls, mk, mf = ST[lab]
        d = F[F.method.map(NAME) == lab].set_index("shape").reindex(shapes)
        axs[0].plot(range(4), d.MRR, color=c, ls=ls, marker=mk, mfc=mf, ms=4, lw=1.1, label=lab)
        e = S[S.method.map(NAME) == lab].set_index("scenario")
        axs[1].plot(range(4), e.reindex(scen).MRR.values, color=c, ls=ls, marker=mk, mfc=mf, ms=4, lw=1.1)
        axs[1].plot([-0.35], [e.loc["none", "MRR"]], marker=mk, color=c, mfc=mf, ms=4, ls="none", clip_on=False)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["Step", "10", "30", "60"])
    axs[0].set_xlabel("Minutes for the fault to reach 4$\\sigma$")
    axs[0].set_ylabel("MRR"); axs[0].set_ylim(0.4, 1.02); axs[0].set_title("(a) Injected faults, pooled (120)", fontsize=8)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["2", "10", "20", "60"])
    axs[1].set_xlabel("Age of a 10$\\sigma$ drift step (min)"); axs[1].set_ylim(0, 0.5); axs[1].set_xlim(-0.95, 3.3)
    axs[1].set_title("(b) Drift near onset, HAI (58)", fontsize=8)
    axs[1].annotate("no drift", xy=(-0.35, 0.47), fontsize=6.5, ha="center")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, fontsize=7, loc="upper center", handlelength=2.2,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(f"{OUT}/fig_tradeoff.pdf"); plt.close(fig)


def fig_case():
    """Fig. 4: SWaT attack 8 (DPIT301). Needs the prepared SWaT data and cache/swat_models.pkl."""
    import data as Dm
    from recon import ReconModel, greedy, _cum, _wmean
    from scipy.stats import rankdata
    ds = Dm.load("swat")
    nm = pickle.load(open(f"{CACHE}/swat_models.pkl", "rb"))["nm"]
    rm = ReconModel().fit(ds.Xn, nm, "T2")
    a = ds.attacks[ds.attacks.label == "8"].iloc[0]
    X = ds.streams["attack"].X
    j = ds.tags.index("DPIT301")
    upd = np.arange(a.onset - 300, a.stop + 300, 3)
    S1, Q, _ = rm.window(X, upd, True, True)
    S0, _, _ = rm.window(X, upd, False, True)
    Z = (X - nm.mu) / nm.sd
    C = _cum(Z); zf = _wmean(C, upd + 1 - 10, upd + 1)
    exp = []
    for t in range(len(upd)):                     # expected DPIT301 after drift correction
        z = zf[t].copy()
        if len(Q[t]):
            zs = _wmean(C, np.array([upd[t] - 1800]), np.array([upd[t] - 300]))[0]
            _, Qt, fQ, _ = greedy(zs, rm.M, 10, rm.thr_slow); z[Qt] -= fQ
        b = rm.M @ z
        exp.append(nm.mu[j] + nm.sd[j] * (z[j] - b[j] / rm.M[j, j]))
    rank = lambda S: np.array([rankdata(-s, method="average")[j] for s in S])  # noqa: E731
    Dz = np.abs(Z[upd])
    tmin = (upd - a.onset) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True, gridspec_kw=dict(hspace=0.12))
    for ax in (ax1, ax2):
        ax.axvspan(0, (a.stop - a.onset) / 60.0, color="#f9dccf", lw=0)
    ax1.plot(tmin, X[upd, j], color=INK, lw=1.0, label="Reported DPIT301")
    ax1.plot(tmin, exp, color=BLUE, lw=0.9, ls="--", label="Expected (reconstructed)")
    ax1.set_ylabel("Pressure (kPa)"); ymax = float(np.nanmax(X[upd, j])) * 1.12
    ax1.set_ylim(0, ymax * 1.15)
    ax1.legend(loc="upper left", ncol=2, handlelength=2.0, columnspacing=1.0, bbox_to_anchor=(0, 1.02), fontsize=7)
    ax1.text(15.6, ymax * 0.62, it("attack"), fontsize=7, ha="right")
    ax2.plot(tmin, rank(S1), color=BLUE, lw=1.4, label="LENS-R")
    ax2.plot(tmin, rank(S0), color=ORANGE, lw=0.9, ls="--", label="w/o drift correction")
    ax2.plot(tmin, rank(Dz), color=GREEN, lw=0.9, label="Raw deviation")
    ax2.set_yscale("log"); ax2.invert_yaxis(); ax2.set_ylim(55, 0.8)
    ax2.set_yticks([1, 3, 10, 51]); ax2.set_yticklabels(["1", "3", "10", "51"])
    ax2.set_ylabel("Rank of DPIT301"); ax2.set_xlabel("Minutes from attack onset")
    ax2.legend(loc="lower left", ncol=3, handlelength=1.8, columnspacing=0.8, bbox_to_anchor=(0, -0.62), fontsize=7)
    ax2.set_xlim(tmin[0], tmin[-1])
    fig.savefig(f"{OUT}/fig_case.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    fig_method(); fig_accuracy(); fig_tradeoff()
    if _CASE:
        fig_case()
    print(f"figures written to {OUT}")
