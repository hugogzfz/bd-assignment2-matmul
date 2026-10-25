#!/usr/bin/env python3
"""
Analysis of the raw measurements of Assignment 2: summary CSVs, LaTeX tables and figures.

Input : results/runs.csv, results/raw_measurements.csv (written by run_benchmarks.py), data/*.csr (flop counts)
Output: results/summary_*.csv, report/tables/*.tex, figures/fig*.pdf|png

Statistics: median and inter-quartile range (Q3 - Q1, linear interpolation as NumPy's default) of the measured
samples of one run (warm-up samples are excluded). Speed-up = median(baseline) / median(method) at the same n.
Nothing in this script measures anything; every number comes from the CSV files.
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sstats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_inputs as gi  # noqa: E402

RES, FIG, TAB, DATA = ROOT / "results", ROOT / "figures", ROOT / "report" / "tables", ROOT / "data"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "figure.dpi": 110,
                     "savefig.bbox": "tight", "axes.axisbelow": True})
INK, MUTED = "#0b0b0b", "#52514e"
COL = {"ijk": "#2a78d6", "ikj": "#eb6834", "blocked": "#1baf7a", "csr": "#eda100"}   # categorical slots 1-4
MK = {"ijk": "o", "ikj": "s", "blocked": "^", "csr": "D"}
LAB = {"ijk": "baseline i-j-k", "ikj": "i-k-j (loop reordering)", "blocked": "blocked i-k-j (tile 96)",
       "csr": "CSR x CSR (sparse)"}
BS_MAIN = 96
NCOL = {500: "#2a78d6", 1000: "#eb6834", 2000: "#1baf7a"}
U = 2.0 ** -53


def gamma(k):
    return k * U / (1.0 - k * U)


# ----------------------------------------------------------------------------- formatting helpers
def tex(s):
    s = str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    s = s.replace("#", "\\#").replace("->", "$\\rightarrow$")
    return s


def sig(x, k=3):
    """k significant digits, fixed notation (no exponent) for the magnitudes that occur here."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    if x == 0:
        return "0"
    d = k - 1 - int(math.floor(math.log10(abs(x))))
    d = max(d, 0)
    return ("%." + str(d) + "f") % x


def pct(x, k=3):
    return sig(x, k)


def save_fig(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=150)
    plt.close(fig)


def write_table(name, header, rows, colspec, extra=""):
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / (name + ".tex")).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- loading
def load():
    runs = pd.read_csv(RES / "runs.csv", keep_default_na=False, na_values=[""])
    raw = pd.read_csv(RES / "raw_measurements.csv", keep_default_na=False, na_values=[""])
    runs["structure"] = runs["structure"].fillna("")
    raw["structure"] = raw["structure"].fillna("")
    meas = raw[raw.phase == "measured"]
    g = meas.groupby("run_id")
    st = pd.DataFrame({
        "median": g.time_per_mult_s.median(),
        "q1": g.time_per_mult_s.quantile(0.25),
        "q3": g.time_per_mult_s.quantile(0.75),
        "tmin": g.time_per_mult_s.min(),
        "tmax": g.time_per_mult_s.max(),
        "nmeas": g.time_per_mult_s.count(),
        "conv_median": g.conv_s.median(),
    })
    runs = runs.merge(st, left_on="run_id", right_index=True, how="left")
    runs["iqr"] = runs.q3 - runs.q1
    runs["rel_iqr"] = runs.iqr / runs["median"]
    runs["vmhwm_mib"] = runs.vmhwm_kib / 1024.0
    return runs, raw


def ok(runs, **kw):
    r = runs[runs.status == "ok"]
    for k, v in kw.items():
        r = r[r[k] == v]
    return r


def spread_label(r):
    return "%s (%s)" % (sig(r["median"]), sig(r["iqr"]))


# ----------------------------------------------------------------------------- Part A: dense
def dense_part(runs):
    d = runs[runs.campaign == "dense"]
    if d.empty:
        return
    okd = d[d.status == "ok"].copy()
    base = okd[okd.config == "ijk"].set_index("n")["median"]
    okd["speedup_vs_ijk"] = okd.apply(lambda r: base.get(r.n, np.nan) / r["median"], axis=1)
    okd["ns_per_madd"] = okd["median"] / okd.n ** 3 * 1e9
    cols = ["config", "n", "nmeas", "median", "q1", "q3", "tmin", "tmax", "rel_iqr", "ns_per_madd", "speedup_vs_ijk",
            "vmhwm_mib", "correctness", "sha256_C"]
    okd[cols].sort_values(["n", "config"]).to_csv(RES / "summary_dense.csv", index=False)

    # ---- Table: runtime
    sizes = sorted(d.n.unique())
    rows = []
    for n in sizes:
        row = [str(n)]
        reps = []
        for m in ("ijk", "ikj", "blocked"):
            r = d[(d.config == m) & (d.n == n)]
            if r.empty:
                row.append("--"); reps.append("-"); continue
            r = r.iloc[0]
            if r.status == "ok":
                row.append(spread_label(r)); reps.append(str(int(r.nmeas)))
            elif r.status == "not_attempted":
                row.append("\\emph{n/a}"); reps.append("-")
            else:
                row.append("\\emph{%s}" % r.status.replace("_", " ")); reps.append("-")
        for m in ("ikj", "blocked"):
            r = okd[(okd.config == m) & (okd.n == n)]
            row.append(sig(r.speedup_vs_ijk.iloc[0], 3) if not r.empty and not math.isnan(r.speedup_vs_ijk.iloc[0]) else "--")
        row.append("/".join(reps))
        rows.append(row)
    write_table("dense_time", ["$n$", "i-j-k", "i-k-j", "blocked", "$S_{ikj}$", "$S_{blk}$", "reps"], rows, "rrrrrrr")

    # ---- Table: limits
    rows = []
    lim = []
    for m in ("ijk", "ikj", "blocked"):
        r = d[d.config == m]
        okn = r[r.status == "ok"].n
        att = r[r.status.isin(["ok", "timeout", "memory_limit", "error"])].n
        fails = r[r.status.isin(["timeout", "memory_limit", "error"])]
        fs = ", ".join("%d: %s" % (x.n, x.status.replace("_", " ")) for x in fails.itertuples()) or "none"
        na = r[r.status == "not_attempted"].n
        t_at_max = r[(r.status == "ok") & (r.n == okn.max())]
        rows.append([LAB[m].replace("_", "\\_") if False else tex(LAB[m]), str(int(okn.max())) if len(okn) else "--",
                     str(int(att.max())) if len(att) else "--", tex(fs),
                     sig(t_at_max["median"].iloc[0]) if len(t_at_max) else "--",
                     str(int(t_at_max.nmeas.iloc[0])) if len(t_at_max) else "--",
                     ", ".join(str(int(x)) for x in sorted(na)) or "none"])
        lim.append(dict(method=m, largest_feasible=int(okn.max()) if len(okn) else None,
                        largest_attempted=int(att.max()) if len(att) else None, failures=fs))
    pd.DataFrame(lim).to_csv(RES / "summary_limits_dense.csv", index=False)
    write_table("dense_limits", ["Method", "Largest feasible $n$", "Largest attempted $n$", "Failures (limit 300 s)",
                                 "Time at largest ok (s)", "Reps", "Not attempted"], rows, "lrrlrrl")

    # ---- Exponent fits
    fits = []
    for m in ("ijk", "ikj", "blocked"):
        for lab, lo, hi in (("100-500", 100, 500), ("1000-3000", 1000, 3000), ("1000-max", 1000, 10 ** 9)):
            r = okd[(okd.config == m) & (okd.n >= lo) & (okd.n <= hi)]
            if r.n.nunique() >= 3:
                lr = sstats.linregress(np.log(r.n), np.log(r["median"]))
                tcrit = sstats.t.ppf(0.975, len(r) - 2)
                fits.append(dict(method=m, window=lab, n_points=len(r), k=lr.slope, ci95=tcrit * lr.stderr))
    pd.DataFrame(fits).to_csv(RES / "summary_fit_dense.csv", index=False)
    rows = []
    for m in ("ijk", "ikj", "blocked"):
        row = [tex(LAB[m])]
        for lab in ("100-500", "1000-3000", "1000-max"):
            f = [x for x in fits if x["method"] == m and x["window"] == lab]
            row.append("$%.2f \\pm %.2f$ (%d)" % (f[0]["k"], f[0]["ci95"], f[0]["n_points"]) if f else "--")
        rows.append(row)
    write_table("dense_fit", ["Method", "$n=100$--$500$", "$n=1000$--$3000$", "$n\\ge1000$ (all feasible)"], rows, "lccc")

    # ---- Memory
    mm = []
    rows = []
    for m in ("ijk", "ikj", "blocked"):
        r = okd[okd.config == m].sort_values("n")
        if r.empty:
            continue
        b = r[r.n == r.n.min()].vmhwm_mib.iloc[0]
        big = r[r.n >= 1000]
        slope = np.polyfit(big.n ** 2.0, big.vmhwm_mib * 1024 * 1024, 1)[0] if len(big) >= 2 else np.nan
        top = r.iloc[-1]
        mm.append(dict(method=m, baseline_mib_at_nmin=b, nmin=int(r.n.min()), bytes_per_n2=slope,
                       n_max=int(top.n), vmhwm_mib_nmax=top.vmhwm_mib))
        rows.append([tex(LAB[m]), sig(b), sig(slope, 4), str(int(top.n)), sig(top.vmhwm_mib, 4), sig(24.0 * top.n ** 2 / 2 ** 20, 4)])
    pd.DataFrame(mm).to_csv(RES / "summary_memory_dense.csv", index=False)
    write_table("dense_mem", ["Method", "Peak RSS at $n=100$ (MiB)", "Growth (bytes per $n^2$, $n\\ge1000$)", "Largest $n$",
                              "Peak RSS there (MiB)", "$24n^2$ model (MiB)"], rows, "lrrrrr")

    # ---- Figures
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in ("ijk", "ikj", "blocked"):
        r = okd[okd.config == m].sort_values("n")
        ax.fill_between(r.n, r.q1, r.q3, color=COL[m], alpha=0.25, lw=0)
        ax.plot(r.n, r["median"], marker=MK[m], ms=4, lw=1.4, color=COL[m], label=LAB[m])
        for x in d[(d.config == m) & (d.status == "timeout")].itertuples():
            ax.plot([x.n], [T_LIMIT], marker="x", ms=8, color=COL[m], mew=2)
    ref = okd[(okd.config == "ijk") & (okd.n == 500)]
    if not ref.empty:
        nn = np.array([100, 8000])
        ax.plot(nn, ref["median"].iloc[0] * (nn / 500.0) ** 3, ls="--", color="#8a8984", lw=0.9)
        ax.text(150, ref["median"].iloc[0] * (150 / 500.0) ** 3 * 2.2, "$n^3$ (anchored at $n=500$)", color=MUTED, fontsize=7.5, rotation=32)
    ax.axhline(T_LIMIT, color="#8a8984", lw=0.8, ls=":")
    ax.text(105, T_LIMIT * 1.12, "300 s limit (x = timeout)", color=MUTED, fontsize=7.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("matrix size $n$"); ax.set_ylabel("time per multiplication (s)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig1_dense_runtime")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ("ijk", "ikj", "blocked"):
        r = okd[okd.config == m].sort_values("n")
        ax.plot(r.n, r.ns_per_madd, marker=MK[m], ms=4, lw=1.4, color=COL[m], label=LAB[m])
    ax.set_xscale("log"); ax.set_ylim(bottom=0)
    ax.set_xlabel("matrix size $n$"); ax.set_ylabel("time per multiply--add, $t/n^3$ (ns)")
    ax.legend(frameon=False)
    save_fig(fig, "fig2_dense_normalised")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ("ikj", "blocked"):
        r = okd[okd.config == m].sort_values("n")
        ax.plot(r.n, r.speedup_vs_ijk, marker=MK[m], ms=4, lw=1.4, color=COL[m], label=LAB[m])
    ax.axhline(1.0, color="#8a8984", lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(bottom=0)
    ax.set_xlabel("matrix size $n$"); ax.set_ylabel("speed-up over the baseline (median ratio)")
    ax.legend(frameon=False)
    save_fig(fig, "fig3_dense_speedup")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ("ijk", "ikj", "blocked"):
        r = okd[okd.config == m].sort_values("n")
        ax.plot(r.n, r.vmhwm_mib, marker=MK[m], ms=4, lw=1.4, color=COL[m], label=LAB[m] + " (VmHWM)")
    nn = np.array([100, 8000])
    ax.plot(nn, 24.0 * nn ** 2 / 2 ** 20, ls="--", color="#8a8984", lw=0.9, label="$24n^2$ bytes (three matrices)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("matrix size $n$"); ax.set_ylabel("peak resident memory (MiB)")
    ax.legend(frameon=False, fontsize=7.5)
    save_fig(fig, "fig4_dense_memory")


T_LIMIT = 300.0


# ----------------------------------------------------------------------------- tile-size sweep, flags, variability
def aux_dense(runs, raw):
    b = runs[runs.campaign.isin(["bsweep", "bsweep2"]) & (runs.status == "ok")]
    if not b.empty:
        piv = b.pivot_table(index="block_size", columns="campaign", values="median")
        piv["mean"] = piv.mean(axis=1)
        piv.to_csv(RES / "summary_bsweep.csv")
        rows = []
        best = piv["mean"].idxmin()
        for bs, r in piv.iterrows():
            rows.append([("\\textbf{%d}" % bs) if bs == best else str(int(bs)), sig(r.get("bsweep")), sig(r.get("bsweep2")),
                         ("\\textbf{%s}" % sig(r["mean"])) if bs == best else sig(r["mean"]),
                         sig(3 * 8 * bs * bs / 1024.0, 3)])
        write_table("bsweep", ["Tile $b$", "Process 1 (s)", "Process 2 (s)", "Mean (s)", "$3\\cdot8b^2$ (KiB)"], rows, "rrrrr")
        fig, ax = plt.subplots(figsize=(6.0, 3.3))
        for c, mk in (("bsweep", "o"), ("bsweep2", "s")):
            if c in piv:
                ax.plot(piv.index, piv[c], marker=mk, ms=4, lw=0, color=COL["blocked"], alpha=0.7,
                        label="independent process %d" % (1 if c == "bsweep" else 2))
        ax.plot(piv.index, piv["mean"], lw=1.4, color=COL["blocked"], label="mean of the two medians")
        ax.axvline(best, color="#8a8984", lw=0.8, ls=":")
        ax.set_xscale("log", base=2); ax.set_xlabel("tile size $b$ (elements)")
        ax.set_ylabel("time at $n=1500$ (s)")
        ax.axvspan(32 / 2, 32, color="none")
        ax.legend(frameon=False, fontsize=7.5)
        save_fig(fig, "fig5_tile_sweep")

    f = runs[(runs.campaign == "flags") & (runs.status == "ok")]
    if not f.empty:
        f = f.copy()
        f["variant"] = f.config.str.split("-", n=1).str[1]
        f["m"] = f.method
        piv = f.pivot_table(index=["n", "m"], columns="variant", values="median")
        piv.to_csv(RES / "summary_flags.csv")
        rows = []
        for (n, m), r in piv.iterrows():
            base = r.get("O2", np.nan)
            row = [str(n), tex(m)]
            for v in ("O2", "O2-unroll", "O3", "O3-native"):
                x = r.get(v, np.nan)
                row.append("%s (%.2f$\\times$)" % (sig(x), base / x) if v != "O2" and not math.isnan(x) else sig(x))
            rows.append(row)
        write_table("flags", ["$n$", "method", "\\texttt{-O2} (s)", "\\texttt{-O2 -funroll-loops}", "\\texttt{-O3}",
                              "\\texttt{-O3 -march=native}"], rows, "rlrrrr")
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
        vc = {"O2": "#2a78d6", "O2-unroll": "#eb6834", "O3": "#1baf7a", "O3-native": "#eda100"}
        for ax, n in zip(axes, sorted(f.n.unique())):
            ms = ["ijk", "ikj", "blocked"]
            for i, v in enumerate(["O2", "O2-unroll", "O3", "O3-native"]):
                vals = [piv.loc[(n, m)].get(v, np.nan) if (n, m) in piv.index else np.nan for m in ms]
                ax.bar(np.arange(3) + (i - 1.5) * 0.2, vals, 0.18, color=vc[v], label=v if n == sorted(f.n.unique())[0] else None)
            ax.set_xticks(range(3)); ax.set_xticklabels(["i-j-k", "i-k-j", "blocked"])
            ax.set_title("$n=%d$" % n); ax.grid(axis="x", visible=False)
            ax.set_ylabel("time per multiplication (s)")
        axes[0].legend(frameon=False, fontsize=7.5, title="gcc options")
        save_fig(fig, "fig6_compiler_flags")

    # between-process variability at n = 1000
    pv = runs[(runs.status == "ok") & (runs.n == 1000) & (runs.structure == "") &
              (runs.campaign.str.startswith("procvar") | (runs.campaign == "dense"))]
    if not pv.empty:
        rows, recs = [], []
        for m in ("ijk", "ikj", "blocked"):
            r = pv[pv.config == m].sort_values("campaign")
            meds = r["median"].values
            recs.append(dict(method=m, processes=len(meds), min=meds.min(), max=meds.max(), median_of_medians=np.median(meds),
                             spread_pct=100 * (meds.max() - meds.min()) / np.median(meds),
                             within_iqr_pct_median=100 * np.median(r.rel_iqr.values)))
            rows.append([tex(LAB[m]), str(len(meds))] + [sig(x) for x in meds[:6]] + [sig(recs[-1]["spread_pct"], 3),
                                                                                     sig(recs[-1]["within_iqr_pct_median"], 3)])
        pd.DataFrame(recs).to_csv(RES / "summary_procvar.csv", index=False)
        write_table("procvar", ["Method", "Proc."] + ["P%d (s)" % i for i in range(1, 7)] + ["Spread (\\%)", "Within IQR (\\%)"],
                    [r + ["--"] * (10 - len(r)) if len(r) < 10 else r for r in rows], "lrrrrrrrrr")

    d = runs[(runs.campaign == "dense") & (runs.status == "ok")]
    if not d.empty:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for m in ("ijk", "ikj", "blocked"):
            r = d[d.config == m].sort_values("n")
            ax.plot(r.n, 100 * r.rel_iqr, marker=MK[m], ms=4, lw=1.2, color=COL[m], label=LAB[m])
        ax.set_xscale("log"); ax.set_ylim(bottom=0)
        ax.set_xlabel("matrix size $n$"); ax.set_ylabel("IQR / median within a process (%)")
        ax.legend(frameon=False, fontsize=7.5)
        save_fig(fig, "fig10_variability")


# ----------------------------------------------------------------------------- bitwise identity
def bitwise(runs):
    r = runs[(runs.status == "ok") & runs.sha256_C.notna() & (runs.sha256_C != "")]
    r = r.copy()
    r["input"] = r.apply(lambda x: "dense-uniform n=%d" % x.n if x.structure == "" else "%s n=%d d=%g%%" % (x.structure, x.n, x.density_pct), axis=1)
    g = r.groupby("input").agg(n_runs=("run_id", "count"), distinct_sha256=("sha256_C", "nunique"),
                               methods=("config", lambda s: ",".join(sorted(set(s)))))
    g.to_csv(RES / "summary_bitwise.csv")
    return g


# ----------------------------------------------------------------------------- Part B: sparse
_flop_cache = {}


def flops_of(n, struct, d):
    key = (n, struct, d)
    if key not in _flop_cache:
        _, pa, ia, _ = gi.read_csr(str(DATA / gi.sparse_name(0, struct, n, d, "csr")))
        _, pb, _, _ = gi.read_csr(str(DATA / gi.sparse_name(1, struct, n, d, "csr")))
        _flop_cache[key] = int((np.bincount(ia, minlength=n).astype(np.int64) * np.diff(pb)).sum())
    return _flop_cache[key]


def sparse_part(runs):
    s = runs[(runs.campaign == "sparse")]
    if s.empty:
        return
    recs = []
    for (n, struct, dens), g in s[s.status == "ok"].groupby(["n", "structure", "density_pct"]):
        row = dict(n=n, structure=struct, density_pct=dens)
        for m in ("csr", "ikj", "blocked", "ijk"):
            r = g[g.config == m]
            if not r.empty:
                r = r.iloc[0]
                row["t_" + m] = r["median"]; row["iqr_" + m] = r.iqr; row["reps_" + m] = int(r.nmeas)
                row["rss_" + m] = r.vmhwm_mib; row["ok_" + m] = r.correctness
                if m == "csr":
                    row.update(nnz_a=r.nnz_a, nnz_b=r.nnz_b, nnz_c=r.nnz_c, t_conv=r.conv_median)
        row["flops"] = flops_of(int(n), struct, float(dens))
        recs.append(row)
    df = pd.DataFrame(recs)
    df["out_density_pct"] = 100.0 * df.nnz_c / df.n ** 2
    df["in_density_actual_pct"] = 100.0 * df.nnz_a / df.n ** 2
    df["t_dense_best"] = df[["t_ikj", "t_blocked"]].min(axis=1)
    df["S_kernel_vs_best"] = df.t_dense_best / df.t_csr
    df["S_total_vs_best"] = df.t_dense_best / (df.t_csr + df.t_conv)
    df["S_kernel_vs_ijk"] = df.t_ijk / df.t_csr
    df["ns_per_flop_csr"] = df.t_csr / df.flops * 1e9
    df["conv_share_pct"] = 100 * df.t_conv / (df.t_conv + df.t_csr)
    df = df.sort_values(["n", "structure", "density_pct"])
    df.to_csv(RES / "summary_sparse.csv", index=False)

    # break-even densities
    be = []
    for (n, struct), g in df.groupby(["n", "structure"]):
        g = g.sort_values("density_pct")
        for label, col in (("kernel only", "S_kernel_vs_best"), ("kernel + conversion", "S_total_vs_best")):
            x, y = g.density_pct.values.astype(float), g[col].values
            cross = None
            for i in range(len(x) - 1):
                if y[i] >= 1.0 > y[i + 1]:
                    t = (math.log(y[i]) - 0.0) / (math.log(y[i]) - math.log(y[i + 1]))
                    cross = math.exp(math.log(x[i]) + t * (math.log(x[i + 1]) - math.log(x[i])))
                    break
            be.append(dict(n=n, structure=struct, boundary=label, breakeven_density_pct=cross,
                           sparse_faster_at_all_tested=bool((y >= 1).all()), sparse_slower_at_all_tested=bool((y < 1).all()),
                           max_density_tested=float(x.max()), speedup_at_max_density=float(y[-1])))
    pd.DataFrame(be).to_csv(RES / "summary_breakeven.csv", index=False)

    # tables: one per size (n = 1000, 2000 in the report; all in CSV)
    for n in sorted(df.n.unique()):
        rows = []
        for struct in ("unif", "band"):
            g = df[(df.n == n) & (df.structure == struct)]
            if g.empty:
                continue
            for r in g.itertuples():
                rows.append([("random" if struct == "unif" else "band") if r.density_pct == g.density_pct.min() else "",
                             pct(float(r.density_pct)), sig(r.in_density_actual_pct, 3), sig(r.out_density_pct, 3),
                             sig(r.t_csr), sig(r.t_conv), sig(r.t_dense_best), sig(r.t_ijk) if not math.isnan(r.t_ijk) else "--",
                             sig(r.S_kernel_vs_best, 3), sig(r.S_total_vs_best, 3)])
            rows.append("MID")
        rows = rows[:-1] if rows and rows[-1] == "MID" else rows
        write_table("sparse_n%d" % n, ["Input", "$d$ (\\%)", "actual (\\%)", "$C$ (\\%)", "CSR kernel (s)", "conv. (s)",
                                       "best dense (s)", "i-j-k (s)", "$S_k$", "$S_t$"], rows, "lrrrrrrrrr")

    # figure: time vs density
    sizes = [x for x in (1000, 2000) if x in set(df.n)]
    fig, axes = plt.subplots(len(sizes), 2, figsize=(7.0, 3.0 * len(sizes)), squeeze=False, sharey="row")
    for i, n in enumerate(sizes):
        for j, struct in enumerate(("unif", "band")):
            ax = axes[i][j]
            g = df[(df.n == n) & (df.structure == struct)].sort_values("density_pct")
            x = g.density_pct.astype(float)
            ax.plot(x, g.t_csr, marker=MK["csr"], ms=4, lw=1.4, color=COL["csr"], label="CSR kernel")
            ax.plot(x, g.t_csr + g.t_conv, marker=MK["csr"], ms=3, lw=1.0, ls="--", color=COL["csr"], mfc="none",
                    label="CSR kernel + dense$\\to$CSR conversion")
            for m in ("ikj", "blocked", "ijk"):
                if "t_" + m in g and g["t_" + m].notna().any():
                    ax.plot(x[g["t_" + m].notna()], g["t_" + m].dropna(), marker=MK[m], ms=4, lw=1.2, color=COL[m], label=LAB[m])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title("n=%d, %s nonzeros" % (n, "uniformly random" if struct == "unif" else "banded"))
            if i == len(sizes) - 1:
                ax.set_xlabel("input density $d$ (% nonzero entries)")
            if j == 0:
                ax.set_ylabel("time per multiplication (s)")
    axes[0][0].legend(frameon=False, fontsize=6.8, loc="lower right")
    save_fig(fig, "fig7_sparse_vs_dense")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axes[0]
    for n in sorted(df.n.unique()):
        for struct, ls in (("unif", "-"), ("band", "--")):
            g = df[(df.n == n) & (df.structure == struct)].sort_values("density_pct")
            ax.plot(g.density_pct.astype(float), g.out_density_pct, ls=ls, marker="o", ms=3, lw=1.2, color=NCOL.get(n, "#6250d6"),
                    label="n=%d %s" % (n, "random" if struct == "unif" else "band"))
    ax.plot([0.1, 90], [0.1, 90], color="#8a8984", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("input density $d$ (%)"); ax.set_ylabel("density of the product $C$ (%)")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for n in sorted(df.n.unique()):
        for struct, ls in (("unif", "-"), ("band", "--")):
            g = df[(df.n == n) & (df.structure == struct)].sort_values("density_pct")
            ax.plot(g.density_pct.astype(float), g.S_kernel_vs_best, ls=ls, marker="o", ms=3, lw=1.2, color=NCOL.get(n, "#6250d6"))
    ax.axhline(1.0, color="#8a8984", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("input density $d$ (%)"); ax.set_ylabel("speed-up of CSR kernel over best dense")
    save_fig(fig, "fig8_sparse_speedup_output_density")

    # sparse memory table (n = 2000)
    rows = []
    n = 2000 if 2000 in set(df.n) else int(df.n.max())
    for struct in ("unif", "band"):
        g = df[(df.n == n) & (df.structure == struct)]
        for r in g.itertuples():
            est = (12 * (r.nnz_a + r.nnz_b + r.nnz_c) + 3 * 8 * (n + 1)) / 2 ** 20
            rows.append([("random" if struct == "unif" else "band") if r.density_pct == g.density_pct.min() else "",
                         pct(float(r.density_pct)), sig(r.out_density_pct, 3), sig(est, 3), sig(r.rss_csr, 3),
                         sig(r.rss_ikj, 3) if not math.isnan(r.rss_ikj) else "--"])
        rows.append("MID")
    if rows:
        rows = rows[:-1]
        write_table("sparse_mem", ["Input", "$d$ (\\%)", "$C$ (\\%)", "CSR storage of $A,B,C$ (MiB)", "peak RSS sparse (MiB)",
                                   "peak RSS dense (MiB)"], rows, "lrrrrr")


# ----------------------------------------------------------------------------- sparse scaling
def scale_part(runs):
    s = runs[runs.campaign == "scale"].copy()
    if s.empty:
        return
    s = s.sort_values(["density_pct", "n"])
    s["flops"] = np.nan
    for i, r in s.iterrows():
        if r.status == "ok":
            s.loc[i, "flops"] = flops_of(int(r.n), "unif", float(r.density_pct))
    s["out_density_pct"] = 100.0 * s.nnz_c / s.n ** 2
    s.to_csv(RES / "summary_scale.csv", index=False)
    rows = []
    for dens, g in s.groupby("density_pct"):
        for r in g.itertuples():
            if r.status == "ok":
                est = (12 * (r.nnz_a + r.nnz_b + r.nnz_c) + 3 * 8 * (r.n + 1)) / 2 ** 20
                rows.append([pct(float(dens)) if r.n == g.n.min() else "", str(int(r.n)), "%.3g" % r.nnz_a, "%.3g" % r.nnz_c,
                             sig(r.out_density_pct, 3), sig(r.median), str(int(r.nmeas)), sig(r.vmhwm_mib, 4), sig(est, 4)])
            else:
                why = r.status.replace("_", " ") if r.status != "not_attempted" else "not attempted"
                rows.append([pct(float(dens)) if r.n == g.n.min() else "", str(int(r.n))] + ["--"] * 3 + ["\\emph{%s}" % why] + ["--"] * 3)
        rows.append("MID")
    rows = rows[:-1]
    write_table("sparse_scale", ["$d$ (\\%)", "$n$", "nnz($A$)", "nnz($C$)", "$C$ dens. (\\%)", "time (s)", "reps", "peak RSS (MiB)",
                                 "CSR est. (MiB)"], rows, "rrrrrrrrr")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for dens, g in s.groupby("density_pct"):
        gg = g[g.status == "ok"]
        c = NCOL[1000] if dens == 0.1 else NCOL[2000]
        axes[0].plot(gg.n, gg["median"], marker="o", ms=4, lw=1.3, color=c, label="d=%g%%" % dens)
        axes[1].plot(gg.n, gg.vmhwm_mib, marker="o", ms=4, lw=1.3, color=c, label="d=%g%%" % dens)
        for r in g[g.status.isin(["timeout", "memory_limit"])].itertuples():
            axes[1].plot([r.n], [4096], marker="x", ms=8, mew=2, color=c)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    for ax_ in axes:
        ax_.xaxis.set_minor_formatter(NullFormatter())
        ax_.xaxis.set_major_locator(FixedLocator([2000, 4000, 8000, 16000, 32000, 64000]))
        ax_.xaxis.set_major_formatter(FixedFormatter(["2k", "4k", "8k", "16k", "32k", "64k"]))
    axes[1].axhline(4096, color="#8a8984", lw=0.8, ls=":")
    axes[1].text(2100, 4300, "4 GiB budget (x = failure)", color=MUTED, fontsize=7.5)
    axes[0].set_xlabel("matrix size $n$"); axes[0].set_ylabel("time per multiplication (s)")
    axes[1].set_xlabel("matrix size $n$"); axes[1].set_ylabel("peak resident memory (MiB)")
    axes[0].legend(frameon=False)
    save_fig(fig, "fig9_sparse_scaling")


# ----------------------------------------------------------------------------- correctness summary
def correctness_part(runs):
    p = RES / "correctness_tests.csv"
    if p.exists():
        t = pd.read_csv(p)
        t["group"] = t.test.str.replace(r"\s*\[.*\]", "", regex=True)
        cat = []
        def count(mask):
            return int(mask.sum()), int((t[mask].result == "pass").sum())
        groups = [
            ("Dense: known answers (scalar, 2$\\times$2, 3$\\times$3, zero, identity, all-ones, integer)",
             t.test.str.contains(r"scalar|2x2|3x3|zero|identity|all-ones|integer") & t.method.isin(["ijk", "ikj", "blocked"])),
            ("Dense: random matrices vs NumPy ($n=5,\\dots,129$)", t.test.str.startswith("random uniform vs NumPy")),
            ("Dense: bit-for-bit equality of all methods and 13 tile sizes", t.test.str.contains(r"bitwise vs ijk|tile-size sweep") & ~t.test.str.contains("csr")),
            ("Dense: repeated runs give identical checksums", t.test.str.contains("repeated runs") & t.method.isin(["ijk", "ikj", "blocked"])),
            ("Sparse: edge cases (empty, identity, diagonal, empty row/column, $1\\times1$, fully dense)",
             t.method.isin(["csr", "csrfile"]) & t.test.str.contains(r"empty|identity|diagonal|1x1|fully dense") & ~t.test.str.contains("repeated")),
            ("Sparse: random and banded vs SciPy (pattern, values), CSR sorted, csr$=$csrfile",
             t.method.isin(["csr", "csrfile"]) & t.test.str.contains(r"unif|band") & ~t.test.str.contains("densified")),
            ("Sparse: densified result bit-for-bit equal to the dense methods", t.test.str.contains("densified")),
            ("Sparse: conversion, spot-check rows, repeated runs", t.test.str.contains(r"conversion|ROWCHK|repeated") & t.method.isin(["csr"])),
        ]
        rows, used = [], pd.Series(False, index=t.index)
        for lab, mask in groups:
            m = mask & ~used
            n, ok_ = count(m)
            used |= m
            rows.append([lab, str(n), str(ok_)])
        rest = ~used
        if rest.any():
            n, ok_ = count(rest)
            rows.append(["Other checks", str(n), str(ok_)])
        rows.append("MID")
        rows.append(["\\textbf{Total}", "\\textbf{%d}" % len(t), "\\textbf{%d}" % int((t.result == "pass").sum())])
        write_table("correctness", ["Group of tests", "Checks", "Passed"], rows, "lrr")
    ok_runs = runs[runs.status == "ok"]
    summ = ok_runs.groupby("campaign").agg(runs=("run_id", "count"), passed=("correctness", lambda s: int((s == "pass").sum())),
                                           max_rel_err=("max_rel_err", "max"))
    summ.to_csv(RES / "summary_validation_runs.csv")


def env_table():
    e = json.load(open(RES / "environment.json"))
    rows = [("CPU", "%s, %d vCPU (KVM guest), single thread used" % (e["cpu_model"], e["logical_processors"])),
            ("Caches", "L1d 32\\,KiB, L2 1\\,MiB per core, L3 33\\,MiB; 64\\,B lines"),
            ("SIMD flags", "SSE4.2, AVX, AVX2, AVX-512F, FMA (available; not used at \\texttt{-O2})"),
            ("RAM / swap", "%.1f\\,GiB / none" % (e["ram_total_MiB"] / 1024)),
            ("OS / kernel", "%s, kernel %s" % (e["os"], e["kernel"])),
            ("Compiler", "gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)"),
            ("Primary flags", "\\texttt{-O2 -std=c11 -ffp-contract=off}"),
            ("Python / packages", "CPython %s; NumPy %s, SciPy %s (outside timed regions)" % (e["python"]["version"], e["python"]["numpy_version_(input_generation_and_reference_only)"], e["python"]["scipy_version_(sparse_reference_only)"])),
            ("Threads", "1 in every kernel; processes not pinned; frequency governor not visible")]
    t = "\\begin{tabular}{@{}lp{11cm}@{}}\n\\toprule\nItem & Value \\\\\n\\midrule\n" + "".join("%s & %s \\\\\n" % r for r in rows) + "\\bottomrule\n\\end{tabular}\n"
    open(TAB / "env.tex", "w").write(t)


def main():
    if not (RES / "runs.csv").exists():
        sys.exit("no results yet")
    runs, raw = load()
    correctness_part(runs)
    dense_part(runs)
    aux_dense(runs, raw)
    bw = bitwise(runs)
    sparse_part(runs)
    scale_part(runs)
    env_table()
    print("bitwise groups: %d, groups with more than one distinct digest: %d" % (len(bw), int((bw.distinct_sha256 > 1).sum())))
    print("analysis written to results/summary_*.csv, report/tables, figures")


if __name__ == "__main__":
    main()
