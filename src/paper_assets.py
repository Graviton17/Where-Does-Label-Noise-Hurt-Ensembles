"""Build every figure and table used in the paper directly from the result files,
so that no number in the paper is typed by hand.

    conda run -n labelnoise python src/paper_assets.py

Writes paper/figures/*.pdf and paper/tables/*.tex
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES, RAW = BASE / "results", BASE / "results" / "raw"
FIG, TAB = BASE / "paper" / "figures", BASE / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["Tree", "Bagging", "RF", "AdaBoost", "GB", "XGB"]
TYPES = ["symmetric", "one_sided", "boundary"]
TYPE_LABEL = {"symmetric": "Symmetric", "one_sided": "One-sided", "boundary": "Boundary"}

# IEEE: figures readable at column width (3.5 in) / text width (7.16 in); 8-pt serif text
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COLORS = {"Tree": "#7f7f7f", "Bagging": "#1f77b4", "RF": "#17becf",
          "AdaBoost": "#d62728", "GB": "#ff7f0e", "XGB": "#9467bd"}
STYLE = {"Tree": ":", "Bagging": "-", "RF": "-", "AdaBoost": "--", "GB": "--", "XGB": "--"}


def deltas(df):
    clean = df[df.noise_type == "none"].set_index(["dataset", "model"])
    noisy = df[df.noise_type != "none"].copy()
    key = list(zip(noisy.dataset, noisy.model))
    for col in ["error", "bias", "net_var"]:
        noisy["d_" + col] = noisy[col].to_numpy() - clean.loc[key, col].to_numpy()
    noisy["clean_error"] = clean.loc[key, "error"].to_numpy()
    return noisy


def load(folder):
    return pd.concat([pd.read_csv(f) for f in sorted((RAW / folder).glob("*.csv"))], ignore_index=True)


main = pd.read_csv(RES / "main_results.csv")
d = deltas(main)


# ------------------------------------------------------------------ Fig. 1: RQ1
def fig_rq1():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.7), sharey=True)
    for ax, nt in zip(axes, TYPES):
        g = d[d.noise_type == nt].groupby(["model", "rho"]).d_error
        for m in MODELS:
            mean = g.mean()[m] * 100
            sem = g.sem()[m] * 100
            x = np.r_[0, mean.index.values * 100]
            y = np.r_[0, mean.values]
            e = np.r_[0, sem.values]
            ax.plot(x, y, STYLE[m], color=COLORS[m], marker="o", ms=2.5, lw=1.1, label=m)
            ax.fill_between(x, y - e, y + e, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_title(TYPE_LABEL[nt] + " noise")
        ax.set_xlabel("Noise rate (%)")
        ax.set_xticks([0, 5, 10, 20, 30])
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Increase in test error (pp)")
    axes[-1].legend(frameon=False, ncol=1, loc="upper left")
    fig.savefig(FIG / "fig_rq1_error_vs_noise.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2: RQ2
def fig_rq2():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.7), sharey=True)
    s = d[np.isclose(d.rho, 0.2)].groupby(["noise_type", "model"])[["d_bias", "d_net_var"]].mean() * 100
    x = np.arange(len(MODELS))
    for ax, nt in zip(axes, TYPES):
        b = s.loc[nt].reindex(MODELS)
        ax.bar(x, b.d_bias, 0.65, color="#4c72b0", label="Δ bias")
        bottom = np.where(b.d_bias > 0, b.d_bias, 0)
        ax.bar(x, b.d_net_var, 0.65, bottom=bottom, color="#dd8452", label="Δ net variance")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, MODELS, rotation=35, ha="right")
        ax.set_title(TYPE_LABEL[nt] + " noise, 20 %")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Change at 20 % noise (pp)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "fig_rq2_bias_variance.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 3: capacity
def fig_capacity():
    cap = deltas(load("capacity"))
    both = pd.concat([d, cap], ignore_index=True)
    depth = {"Bagging_d3": 3, "Bagging": 12, "RF_d3": 3, "RF": 12,
             "AdaBoost": 1, "AdaBoost_d3": 3, "GB": 3, "GB_d8": 8, "XGB_d3": 3, "XGB": 4, "XGB_d8": 8}
    groups = {"Bagging": ["Bagging_d3", "Bagging"], "RF": ["RF_d3", "RF"],
              "AdaBoost": ["AdaBoost", "AdaBoost_d3"], "GB": ["GB", "GB_d8"], "XGB": ["XGB_d3", "XGB", "XGB_d8"]}
    s = both[np.isclose(both.rho, 0.2)].groupby(["noise_type", "model"]).d_error.mean() * 100
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.6), sharey=True)
    for ax, nt in zip(axes, TYPES):
        for fam, ms in groups.items():
            ax.plot([depth[m] for m in ms], [s[(nt, m)] for m in ms], STYLE[fam], marker="o", ms=3,
                    color=COLORS[fam], lw=1.1, label=fam)
        ax.set_xticks([1, 3, 4, 8, 12], ["1", "3", "4", "8", "full"])
        ax.set_xlabel("Maximum tree depth")
        ax.set_title(TYPE_LABEL[nt] + " noise, 20 %")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Increase in test error (pp)")
    axes[1].legend(frameon=False, loc="upper center", ncol=3)
    fig.savefig(FIG / "fig_capacity_depth.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ tables
def fmt(x, nd=1):
    return f"{x:.{nd}f}".replace("-", "$-$")


def tab_datasets():
    t = pd.read_csv(RES / "tables" / "datasets.csv")
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Dataset & ID & Rows & Feat. & Pos. \\", r"\midrule"]
    for r in t.itertuples():
        name = r.dataset.replace("_", "\\_")
        lines.append(f"{name} & {r.openml_id} & {r.rows:,} & "
                     f"{r.features_after_encoding} & {r.positive_share:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_datasets.tex").write_text("\n".join(lines))


def tab_rq2():
    s = d[np.isclose(d.rho, 0.2)].groupby(["noise_type", "model"])[["d_error", "d_bias", "d_net_var"]].mean()
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Model & $\Delta$err & $\Delta$bias & $\Delta$var & Var.\ share \\", r"\midrule"]
    for nt in TYPES:
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{TYPE_LABEL[nt]} noise}}}} \\")
        for m in MODELS:
            r = s.loc[(nt, m)]
            lines.append(f"{m} & {fmt(100 * r.d_error)} & {fmt(100 * r.d_bias)} & {fmt(100 * r.d_net_var)} & "
                         f"{100 * r.d_net_var / r.d_error:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_rq2_split.tex").write_text("\n".join(lines))


def tab_capacity_tests():
    t = pd.read_csv(RES / "tables" / "phase6_capacity_equal_depth_tests.csv")
    t = t[t.noise_type.isin(TYPES)]
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Parallel & Sequential & Sym. & One-s. & Bound. \\", r"\midrule"]
    for (p, q), g in t.groupby(["parallel (d3)", "sequential (d3)"], sort=False):
        cells = []
        for nt in TYPES:
            r = g[g.noise_type == nt].iloc[0]
            star = "$^{*}$" if r["p (Wilcoxon)"] < 0.05 else ""
            cells.append(fmt(r["damage difference (pp), parallel − sequential"]) + star)
        pn, qn = p.replace("_", "\\_"), q.replace("_", "\\_")
        lines.append(f"{pn} & {qn} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_capacity_tests.tex").write_text("\n".join(lines))


def tab_rq4():
    t = pd.read_csv(RES / "tables" / "phase6_rq4_regularisation.csv")
    t = t[np.isclose(t.rho, 0.2)]
    names = {"GB_reg": "GB, early stop", "XGB_reg": "XGB, early stop",
             "AdaBoost_reg": r"AdaBoost, $\eta{=}0.5$", "Bagging_small": "Bagging, 20\\% samples"}
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Fix & Sym. & One-s. & Bound. \\", r"\midrule"]
    for fix, g in t.groupby("fix", sort=False):
        cells = []
        for nt in TYPES:
            r = g[g.noise_type == nt].iloc[0]
            star = "$^{*}$" if r["p (Wilcoxon)"] < 0.05 else ""
            cells.append(f"{r['damage recovered %']:.0f}".replace("-", "$-$") + r"\%" + star)
        lines.append(names[fix] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_rq4_fixes.tex").write_text("\n".join(lines))


def tab_rq3():
    t = pd.read_csv(RES / "tables" / "phase6_rq3_variance_sources.csv", index_col=[0, 1])
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Symmetric} & \multicolumn{2}{c}{One-sided} & \multicolumn{2}{c}{Boundary} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & Noise & Samp. & Noise & Samp. & Noise & Samp. \\", r"\midrule"]
    for m in MODELS:
        cells = []
        for nt in TYPES:
            r = t.loc[(nt, m)]
            cells += [fmt(r["noise adds (A−M)"]), fmt(r["sampling adds (B−M)"])]
        lines.append(m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_rq3_sources.tex").write_text("\n".join(lines))


def tab_margin():
    t = pd.read_csv(RES / "tables" / "phase6_margin_halves.csv", index_col=[0, 1])
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Model & Sym. & One-s. & Bound. \\", r"\midrule"]
    for m in MODELS:
        lines.append(m + " & " + " & ".join(f"{t.loc[(nt, m), 'near / far ratio']:.1f}" for nt in TYPES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_margin.tex").write_text("\n".join(lines))


def tab_ranks():
    """Friedman test + Nemenyi critical difference (Demsar, 2006) on the damage at 20 % noise:
    rank the six models on every dataset (1 = least damaged), average the ranks."""
    from scipy.stats import friedmanchisquare
    q_alpha = {6: 2.850}[len(MODELS)]                 # Nemenyi, alpha = 0.05, k = 6
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Model & Sym. & One-s. & Bound. \\", r"\midrule"]
    ranks, stats = {}, {}
    for nt in TYPES:
        w = d[(d.noise_type == nt) & np.isclose(d.rho, 0.2)].pivot(index="dataset", columns="model",
                                                                    values="d_error")[MODELS]
        ranks[nt] = w.rank(axis=1).mean()
        stats[nt] = friedmanchisquare(*[w[m] for m in MODELS]).pvalue
        n = len(w)
    cd = q_alpha * np.sqrt(len(MODELS) * (len(MODELS) + 1) / (6 * n))
    for m in MODELS:
        lines.append(m + " & " + " & ".join(f"{ranks[nt][m]:.2f}" for nt in TYPES) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Friedman $p$ & " + " & ".join(
        (f"{stats[nt]:.1e}" if stats[nt] < 0.001 else f"{stats[nt]:.3f}") for nt in TYPES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_ranks.tex").write_text("\n".join(lines))
    (TAB / "cd_value.tex").write_text(f"{cd:.2f}")
    pd.DataFrame(ranks).assign(friedman_p=pd.Series(stats)).to_csv(RES / "tables" / "phase7_ranks.csv")
    print("average ranks:\n", pd.DataFrame(ranks).round(2), "\nFriedman p:", stats, "\nCD:", round(cd, 2))


def tab_multiclass():
    """RQ5: damage (pp) and variance share at 20 % noise on the 20 multiclass datasets."""
    t = pd.read_csv(RES / "tables" / "phase6b_split_20pct.csv", index_col=[0, 1])
    types = ["symmetric", "pair_flip", "boundary"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Symmetric} & \multicolumn{2}{c}{Pair-flip} & \multicolumn{2}{c}{Boundary} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & $\Delta$err & Var. & $\Delta$err & Var. & $\Delta$err & Var. \\", r"\midrule"]
    for m in MODELS:
        cells = []
        for nt in types:
            r = t.loc[(nt, m)]
            share = "--" if r["Δ error (pp)"] <= 0 else f"{r['variance share %']:.0f}\\%"
            cells += [fmt(r["Δ error (pp)"]), share]
        lines.append(m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_multiclass.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    tab_multiclass()
    fig_rq1(); fig_rq2(); fig_capacity()
    tab_datasets(); tab_rq2(); tab_capacity_tests(); tab_rq4(); tab_rq3(); tab_margin(); tab_ranks()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print("tables: ", sorted(p.name for p in TAB.glob("*.tex")))
