"""
Statistical tests + plots for the dependency-length comparison.
Reads sentence_results.csv produced by analyze.py.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

NATURAL_C, PROJ_C, RAND_C = "#2563eb", "#f59e0b", "#dc2626"
OUT = Path("./figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("sentence_results.csv")

# ---------- 1. Statistical tests ----------
print("=" * 78)
print("STATISTICAL TESTS (paired Wilcoxon signed-rank, per language)")
print("=" * 78)
print(f"{'Language':<12} {'n':>5} {'Nat<Rand %':>12} {'p (vs Rand)':>14} "
      f"{'Nat<Proj %':>12} {'p (vs Proj)':>14}")
print("-" * 78)

stats_rows = []
for lang in sorted(df["language"].unique()):
    sub = df[df["language"] == lang]
    nat, rand, proj = sub["natural"].values, sub["random"].values, sub["projective"].values

    # Sentence-level comparison: what fraction has natural < randomized?
    pct_nat_below_rand = (nat < rand).mean() * 100
    pct_nat_below_proj = (nat < proj).mean() * 100

    # Wilcoxon signed-rank (one-sided: natural < other)
    w_rand, p_rand = stats.wilcoxon(nat, rand, alternative="less")
    w_proj, p_proj = stats.wilcoxon(nat, proj, alternative="less")

    print(f"{lang:<12} {len(sub):>5} {pct_nat_below_rand:>11.1f}% "
          f"{p_rand:>14.2e} {pct_nat_below_proj:>11.1f}% {p_proj:>14.2e}")

    stats_rows.append({
        "language": lang,
        "n_sentences": len(sub),
        "pct_nat_below_random": pct_nat_below_rand,
        "pct_nat_below_projective": pct_nat_below_proj,
        "p_value_vs_random": p_rand,
        "p_value_vs_projective": p_proj,
    })

pd.DataFrame(stats_rows).to_csv("statistical_tests.csv", index=False)

# ---------- 2. Bar chart: average DL per language ----------
summary = pd.read_csv("language_summary.csv").sort_values("natural")

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(summary))
w = 0.27
ax.bar(x - w, summary["natural"], w, label="Natural", color=NATURAL_C, edgecolor="white")
ax.bar(x, summary["projective"], w, label="Random (projective)", color=PROJ_C, edgecolor="white")
ax.bar(x + w, summary["random"], w, label="Random (any order)", color=RAND_C, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(summary["language"], rotation=30, ha="right")
ax.set_ylabel("Average total dependency length per sentence")
ax.set_title("Dependency length: natural vs randomized word orders\n12 languages, ~2000 sentences each (length 5-50)",
             fontsize=13, pad=14)
ax.legend(loc="upper left", frameon=False)
ax.grid(axis="y", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "01_avg_dl_by_language.png", dpi=140, bbox_inches="tight")
plt.close()
print("\nSaved 01_avg_dl_by_language.png")

# ---------- 3. % reduction (natural vs each baseline) ----------
fig, ax = plt.subplots(figsize=(13, 6))
summary_s = summary.sort_values("nat_vs_proj_pct", ascending=True)
x = np.arange(len(summary_s))
ax.barh(x - 0.2, summary_s["nat_vs_rand_pct"], 0.4, label="vs Random", color=RAND_C)
ax.barh(x + 0.2, summary_s["nat_vs_proj_pct"], 0.4, label="vs Random projective", color=PROJ_C)
for i, (r, p) in enumerate(zip(summary_s["nat_vs_rand_pct"], summary_s["nat_vs_proj_pct"])):
    ax.text(r + 0.6, i - 0.2, f"{r:.1f}%", va="center", fontsize=9, color="#444")
    ax.text(p + 0.6, i + 0.2, f"{p:.1f}%", va="center", fontsize=9, color="#444")
ax.set_yticks(x)
ax.set_yticklabels(summary_s["language"])
ax.set_xlabel("% shorter than baseline (higher = more dependency-length minimization)")
ax.set_title("How much shorter are natural dependency lengths\nthan their randomized counterparts?", fontsize=13, pad=14)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.set_xlim(0, max(summary_s["nat_vs_rand_pct"]) + 8)
plt.tight_layout()
plt.savefig(OUT / "02_pct_reduction.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved 02_pct_reduction.png")

# ---------- 4. DL vs sentence length (averaged in length bins) ----------
df["len_bin"] = pd.cut(df["n"], bins=range(5, 51, 3))
binned = df.groupby("len_bin", observed=True).agg(
    n_mid=("n", "mean"),
    natural=("natural", "mean"),
    projective=("projective", "mean"),
    random=("random", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(binned["n_mid"], binned["natural"], marker="o", lw=2.5, color=NATURAL_C, label="Natural")
ax.plot(binned["n_mid"], binned["projective"], marker="s", lw=2.5, color=PROJ_C, label="Random (projective)")
ax.plot(binned["n_mid"], binned["random"], marker="^", lw=2.5, color=RAND_C, label="Random (any order)")
ax.set_xlabel("Sentence length (tokens)")
ax.set_ylabel("Average total dependency length")
ax.set_title("Dependency length scales differently with sentence length\nfor natural vs randomized word orders (averaged across all 12 languages)",
             fontsize=13, pad=14)
ax.legend(frameon=False, fontsize=11)
ax.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "03_dl_vs_length.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved 03_dl_vs_length.png")

# ---------- 5. Per-language sentence-length-normalized DL distributions ----------
df["natural_norm"] = df["natural"] / df["n"]
df["projective_norm"] = df["projective"] / df["n"]
df["random_norm"] = df["random"] / df["n"]

langs = sorted(df["language"].unique())
fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()
for i, lang in enumerate(langs):
    ax = axes[i]
    sub = df[df["language"] == lang]
    bins = np.linspace(0, 12, 40)
    ax.hist(sub["natural_norm"], bins=bins, alpha=0.6, color=NATURAL_C, label="Natural")
    ax.hist(sub["projective_norm"], bins=bins, alpha=0.5, color=PROJ_C, label="Projective")
    ax.hist(sub["random_norm"], bins=bins, alpha=0.4, color=RAND_C, label="Random")
    ax.set_title(lang, fontsize=11)
    ax.set_xlim(0, 12)
    ax.grid(alpha=0.2, linestyle="--")
axes[0].legend(loc="upper right", fontsize=9, frameon=False)
fig.supxlabel("Dependency length per token", fontsize=12)
fig.supylabel("Number of sentences", fontsize=12)
fig.suptitle("Distribution of per-token dependency length across languages", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "04_distributions.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved 04_distributions.png")

# ---------- 6. Word-order grouping ----------
wo_summary = summary.groupby("word_order")[["natural", "projective", "random"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(wo_summary))
w = 0.27
ax.bar(x - w, wo_summary["natural"], w, label="Natural", color=NATURAL_C)
ax.bar(x, wo_summary["projective"], w, label="Projective", color=PROJ_C)
ax.bar(x + w, wo_summary["random"], w, label="Random", color=RAND_C)
ax.set_xticks(x)
ax.set_xticklabels(wo_summary["word_order"])
ax.set_xlabel("Word order type")
ax.set_ylabel("Average dependency length")
ax.set_title("Dependency length by dominant word order", fontsize=12, pad=14)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT / "05_by_word_order.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved 05_by_word_order.png")

print("\nDone.")
