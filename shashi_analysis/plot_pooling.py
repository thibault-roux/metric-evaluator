"""
plot_pooling.py
---------------
Identifies and visualises the best pooling strategies for the semdist metric
on the HATS dataset.

Produces 4 figures:
  1. Mean accuracy per pooling strategy (all models, certitude=0)
  2. Per-model heatmap: accuracy by model × pooling (certitude=0)
  3. Pooling rank distribution — how often each pooling is #1 per model
  4. Accuracy gain vs last_token baseline (delta plot)

Usage:
    python plot_pooling.py [--csv PATH] [--certitude 0.0] [--out_dir ./plots]
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pooling strategy analysis plots")
parser.add_argument("--csv",        default="summary_results.csv",
                    help="Path to summary_results.csv")
parser.add_argument("--certitude",  type=float, default=0.0,
                    help="Certitude level to use (0.0 | 0.7 | 1.0)")
parser.add_argument("--out_dir",    default="plots_pooling",
                    help="Output directory for saved figures")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
SURFACE   = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
ACCENT    = "#38bdf8"

POOLING_COLORS = {
    "last_token":       "#64748b",
    "mean":             "#38bdf8",
    "weighted_mean":    "#f472b6",
    "penultimate_token": "#a78bfa",
    "last4_mean":       "#34d399",
    "qwen3_eos":        "#fbbf24",
    "mean_no_last_token": "#22d3ee",            # lighter cyan
    "weighted_mean_no_last_token": "#fb7185",   # softer pink/red
}

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         10,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  BORDER,
})

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(args.csv, on_bad_lines="skip")
print(df_raw)
df = df_raw[df_raw["certitude"] == args.certitude].copy()
print(df)
input("Press Enter to continue...")

# Normalise: treat qwen3_eos as a valid pooling name (already in data)
# For non-embedding models last_token == qwen3_embedding, but they appear
# under different column values — no renaming needed, we keep them distinct.

short = lambda s: (s.replace("Qwen/", "")
                    .replace("mistralai/", "")
                    .replace("allenai/", "")
                    .replace("google/", "")
                    )
df["short_name"] = df["model_name"].map(short)

# Ordered poolings for consistent axis
POOLING_ORDER_GEN = [
    "last_token",
    "mean",
    "mean_no_last_token",
    "weighted_mean",
    "weighted_mean_no_last_token",
    "penultimate_token",
    "last4_mean",
]

POOLING_ORDER_EMB = [
    "qwen3_eos",
    "mean",
    "mean_no_last_token",
    "weighted_mean",
    "weighted_mean_no_last_token",
    "penultimate_token",
    "last4_mean",
]

ALL_POOLINGS = [
    "last_token",
    "qwen3_eos",
    "mean",
    "mean_no_last_token",
    "weighted_mean",
    "weighted_mean_no_last_token",
    "penultimate_token",
    "last4_mean",
]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Mean accuracy per pooling strategy
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.suptitle(f"Mean Accuracy per Pooling Strategy  (certitude={args.certitude})",
              fontsize=13, color=TEXT, y=0.98)

print("df:")
print(df)
input("Press Enter to continue...")
pooling_means = (df.groupby("pooling")["accuracy_pct"]
                   .agg(["mean", "std", "count"])
                   .reindex([p for p in ALL_POOLINGS if p in df["pooling"].unique()])
                   .dropna())

bars = ax1.barh(
    pooling_means.index,
    pooling_means["mean"],
    xerr=pooling_means["std"],
    color=[POOLING_COLORS.get(p, "#888") for p in pooling_means.index],
    error_kw=dict(ecolor=MUTED, capsize=4, linewidth=1.2),
    height=0.55,
    zorder=3,
)
# Annotate values
for bar, (_, row) in zip(bars, pooling_means.iterrows()):
    ax1.text(row["mean"] + 0.15, bar.get_y() + bar.get_height() / 2,
             f"{row['mean']:.2f}%", va="center", fontsize=9, color=TEXT)

ax1.set_xlabel("Accuracy %", color=TEXT)
ax1.set_xlim(58, pooling_means["mean"].max() + 3)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax1.invert_yaxis()
ax1.set_title(f"n models per pooling shown in ()", fontsize=9, color=MUTED, pad=6)

# Add count labels on left
for idx, (p, row) in enumerate(pooling_means.iterrows()):
    ax1.text(58.2, idx, f"n={int(row['count'])}", va="center",
             fontsize=8, color=MUTED)

plt.tight_layout()
fig1.savefig(os.path.join(args.out_dir, "fig1_pooling_mean_accuracy.png"),
             dpi=150, bbox_inches="tight")
print(f"[✓] fig1_pooling_mean_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Heatmap: model × pooling accuracy
# ─────────────────────────────────────────────────────────────────────────────
pivot = df.pivot_table(index="short_name", columns="pooling",
                       values="accuracy_pct", aggfunc="first")

# Reorder columns: generative poolings first, then embedding-specific
col_order = [p for p in ALL_POOLINGS if p in pivot.columns]
pivot = pivot[col_order]

# Sort rows by best accuracy across poolings
pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.suptitle(f"Accuracy Heatmap: Model × Pooling  (certitude={args.certitude})",
              fontsize=13, color=TEXT, y=0.99)

sns.heatmap(
    pivot, ax=ax2,
    cmap="YlOrRd",
    annot=True, fmt=".1f",
    linewidths=0.5, linecolor=DARK_BG,
    cbar_kws={"label": "Accuracy %", "shrink": 0.7},
    annot_kws={"size": 9},
)
ax2.set_xlabel("Pooling Strategy", color=TEXT, labelpad=10)
ax2.set_ylabel("Model", color=TEXT, labelpad=10)
ax2.tick_params(axis="x", rotation=25)
ax2.tick_params(axis="y", rotation=0)
ax2.collections[0].colorbar.ax.yaxis.label.set_color(TEXT)
ax2.collections[0].colorbar.ax.tick_params(colors=MUTED)

# Mark best cell per row with a ★
for i, row_name in enumerate(pivot.index):
    best_col = pivot.loc[row_name].idxmax()
    best_j   = list(pivot.columns).index(best_col)
    ax2.text(best_j + 0.5, i + 0.5, "★", ha="center", va="center",
             fontsize=12, color="white", fontweight="bold")

ax2.set_facecolor(SURFACE)
fig2.patch.set_facecolor(DARK_BG)
plt.tight_layout()
fig2.savefig(os.path.join(args.out_dir, "fig2_heatmap_model_pooling.png"),
             dpi=150, bbox_inches="tight")
print(f"[✓] fig2_heatmap_model_pooling.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Pooling rank distribution (how often each pooling is #1 / #2 / #3)
# ─────────────────────────────────────────────────────────────────────────────
ranks = (df.groupby("model_name")
           .apply(lambda g: g.set_index("pooling")["accuracy_pct"]
                             .rank(ascending=False, method="min")
                             .astype(int))
           .reset_index(level=0, drop=True)
           .reset_index())
ranks.columns = ["pooling", "rank"]

max_rank = df["pooling"].nunique()
rank_counts = (ranks.groupby(["pooling", "rank"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=list(range(1, max_rank + 1)), fill_value=0))
                    # .reindex(columns=[1, 2, 3, 4, 5, 6], fill_value=0))
rank_counts = rank_counts.reindex(
    [p for p in ALL_POOLINGS if p in rank_counts.index])

rank_colors = ["#f1c40f", "#bdc3c7", "#cd7f32", "#64748b", "#475569", "#334155"]
rank_labels = ["Rank 1 (best)", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Rank 6"]

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.suptitle(f"Pooling Rank Distribution  (certitude={args.certitude})",
              fontsize=13, color=TEXT)

bottom = np.zeros(len(rank_counts))
for col, color, label in zip(rank_counts.columns, rank_colors, rank_labels):
    vals = rank_counts[col].values.astype(float)
    ax3.barh(rank_counts.index, vals, left=bottom, color=color,
             label=label, height=0.55, zorder=3)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0:
            ax3.text(b + v / 2, i, str(int(v)), ha="center", va="center",
                     fontsize=9, color="black", fontweight="bold")
    bottom += vals

ax3.set_xlabel("Number of models", color=TEXT)
ax3.invert_yaxis()
ax3.legend(loc="lower right", fontsize=9)
ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
fig3.savefig(os.path.join(args.out_dir, "fig3_pooling_rank_distribution.png"),
             dpi=150, bbox_inches="tight")
print(f"[✓] fig3_pooling_rank_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Accuracy delta vs last_token baseline
# ─────────────────────────────────────────────────────────────────────────────
# For each model, compute delta = pooling_acc - last_token_acc
# (embedding models use qwen3_eos as their "native" pooling — use that as baseline)

def get_baseline(group):
    if "last_token" in group["pooling"].values:
        return group.loc[group["pooling"] == "last_token", "accuracy_pct"].values[0]
    elif "qwen3_eos" in group["pooling"].values:
        return group.loc[group["pooling"] == "qwen3_eos", "accuracy_pct"].values[0]
    return None

deltas = []
for model, grp in df.groupby("model_name"):
    baseline = get_baseline(grp)
    if baseline is None:
        continue
    for _, row in grp.iterrows():
        deltas.append({
            "model":   short(model),
            "pooling": row["pooling"],
            "delta":   row["accuracy_pct"] - baseline,
            "variant": row["variant"],
        })

delta_df = pd.DataFrame(deltas)
# Exclude the baseline pooling itself (delta=0) from clutter
baseline_poolings = {"last_token", "qwen3_eos"}
delta_df = delta_df[~delta_df["pooling"].isin(baseline_poolings)]

# Pivot: model × pooling → delta
delta_pivot = delta_df.pivot_table(index="model", columns="pooling",
                                   values="delta", aggfunc="first")
# col_order_d = [p for p in ["mean","weighted_mean","penultimate_token","last4_mean"]
#                if p in delta_pivot.columns]
col_order_d = [
    p for p in [
        "mean",
        "mean_no_last_token",
        "weighted_mean",
        "weighted_mean_no_last_token",
        "penultimate_token",
        "last4_mean",
    ]
    if p in delta_pivot.columns
]
delta_pivot = delta_pivot[col_order_d]
delta_pivot = delta_pivot.loc[delta_pivot.max(axis=1).sort_values(ascending=False).index]

fig4, ax4 = plt.subplots(figsize=(11, 7))
fig4.suptitle(
    f"Accuracy Gain vs Baseline Pooling  (certitude={args.certitude})\n"
    "Baseline = last_token (generative) | qwen3_eos (embedding)",
    fontsize=12, color=TEXT, y=1.0)

sns.heatmap(
    delta_pivot, ax=ax4,
    cmap="RdYlGn", center=0,
    annot=True, fmt="+.2f",
    linewidths=0.5, linecolor=DARK_BG,
    cbar_kws={"label": "Δ Accuracy pp", "shrink": 0.7},
    annot_kws={"size": 9},
) # vmin=-2, vmax=2, center=0,
ax4.set_xlabel("Pooling Strategy", color=TEXT, labelpad=10)
ax4.set_ylabel("Model", color=TEXT, labelpad=10)
ax4.tick_params(axis="x", rotation=20)
ax4.tick_params(axis="y", rotation=0)
ax4.collections[0].colorbar.ax.yaxis.label.set_color(TEXT)
ax4.collections[0].colorbar.ax.tick_params(colors=MUTED)
ax4.set_facecolor(SURFACE)
fig4.patch.set_facecolor(DARK_BG)

plt.tight_layout()
fig4.savefig(os.path.join(args.out_dir, "fig4_pooling_delta_vs_baseline.png"),
             dpi=150, bbox_inches="tight")
print(f"[✓] fig4_pooling_delta_vs_baseline.png")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n── Pooling summary (certitude={}) ──".format(args.certitude))
summary = (df.groupby("pooling")["accuracy_pct"]
             .agg(mean="mean", median="median", std="std", best="max")
             .sort_values("mean", ascending=False))
print(summary.round(2).to_string())
print(f"\nPlots saved to: {os.path.abspath(args.out_dir)}/")
