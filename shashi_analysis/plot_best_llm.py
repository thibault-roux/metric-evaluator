"""
plot_best_llm.py
----------------
Given the pooling analysis (use penultimate_mean / last4_mean for generative
models, qwen3_eos / mean for embedding models), compares all LLMs on their
best pooling configuration.

Produces 5 figures:
  1. Ranked bar — best accuracy per model (best pooling, certitude=0)
  2. Accuracy vs Coverage scatter (all certitude levels, best pooling)
  3. Certitude effect — accuracy gain when filtering low-certitude pairs
  4. Model-size scaling — Qwen3 generative vs embedding families
  5. Base vs Fine-tuned comparison (Qwen3 family)

Usage:
    python plot_best_llm.py [--csv PATH] [--best_gen penultimate_mean]
                             [--best_emb mean] [--out_dir ./plots]

The --best_gen / --best_emb flags let you plug in the winner from plot_pooling.py.
Defaults are set to the findings from that script.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Best-LLM analysis plots")
parser.add_argument("--csv",       default="summary_results.csv")
parser.add_argument("--best_gen",  default="penultimate_mean",
                    help="Best pooling for generative models (from plot_pooling.py)")
parser.add_argument("--best_emb",  default="mean",
                    help="Best pooling for embedding models (from plot_pooling.py)")
parser.add_argument("--out_dir",   default="plots_llm")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"

MODEL_COLORS = {
    "qwen3_base":      "#38bdf8",
    "qwen3_final":     "#0ea5e9",
    "qwen3_embedding": "#a78bfa",
    "mistral":         "#f472b6",
    "olmo2":           "#34d399",
}

CERTITUDE_STYLES = {
    0.0: dict(marker="o", ls="-",  ms=7,  label="certitude=0.0 (full data)"),
    0.7: dict(marker="s", ls="--", ms=7,  label="certitude=0.7"),
    1.0: dict(marker="^", ls=":",  ms=8,  label="certitude=1.0 (high confidence)"),
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

short = lambda s: (s.replace("Qwen/", "")
                    .replace("mistralai/", "")
                    .replace("allenai/", ""))

df_raw["short_name"] = df_raw["model_name"].map(short)

def model_color(row):
    if row["variant"] == "embedding":    return MODEL_COLORS["qwen3_embedding"]
    if row["family"]  == "mistral":      return MODEL_COLORS["mistral"]
    if row["family"]  == "olmo2":        return MODEL_COLORS["olmo2"]
    if row["variant"] == "final":        return MODEL_COLORS["qwen3_final"]
    return MODEL_COLORS["qwen3_base"]

# For each model, pick the "best pooling" row per certitude level
def best_pooling_rows(df_in):
    """Return one row per (model, certitude) using the designated best pooling."""
    out = []
    for (model, cert), grp in df_in.groupby(["model_name", "certitude"]):
        variant = grp["variant"].iloc[0]
        best_pool = args.best_emb if variant == "embedding" else args.best_gen
        row = grp[grp["pooling"] == best_pool]
        if row.empty:
            # Fallback: take best available
            row = grp.loc[[grp["accuracy_pct"].idxmax()]]
        out.append(row.iloc[0])
    return pd.DataFrame(out)

df_best = best_pooling_rows(df_raw)
df_best["color"] = df_best.apply(model_color, axis=1)

# Certitude=0 subset for model ranking
df0 = df_best[df_best["certitude"] == 0.0].sort_values("accuracy_pct", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Ranked bar: best accuracy per model (certitude=0, best pooling)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.suptitle(
    f"Best LLMs — Ranked by Accuracy  (certitude=0.0)\n"
    f"Pooling: generative={args.best_gen} | embedding={args.best_emb}",
    fontsize=12, color=TEXT, y=1.0)

bars = ax1.barh(
    df0["short_name"],
    df0["accuracy_pct"],
    color=df0["color"],
    height=0.6,
    zorder=3,
)
for bar, row in zip(bars, df0.itertuples()):
    ax1.text(row.accuracy_pct + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{row.accuracy_pct:.2f}%",
             va="center", fontsize=9, color=TEXT)

ax1.set_xlabel("Accuracy %", color=TEXT)
ax1.set_xlim(58, df0["accuracy_pct"].max() + 3)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax1.invert_yaxis()

# Legend
legend_patches = [
    mpatches.Patch(color=MODEL_COLORS["qwen3_final"],     label="Qwen3 final (instruct)"),
    mpatches.Patch(color=MODEL_COLORS["qwen3_base"],      label="Qwen3 base"),
    mpatches.Patch(color=MODEL_COLORS["qwen3_embedding"], label="Qwen3 embedding"),
    mpatches.Patch(color=MODEL_COLORS["mistral"],         label="Mistral-7B"),
    mpatches.Patch(color=MODEL_COLORS["olmo2"],           label="OLMo2-7B"),
]
ax1.legend(handles=legend_patches, loc="lower right", fontsize=9)

# Mark #1
ax1.axvline(df0["accuracy_pct"].max(), color="#fbbf24", lw=1, ls="--", alpha=0.5)

plt.tight_layout()
fig1.savefig(os.path.join(args.out_dir, "fig1_llm_ranked_accuracy.png"),
             dpi=150, bbox_inches="tight")
print("[✓] fig1_llm_ranked_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Accuracy vs Coverage scatter (all certitude levels)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.suptitle(
    "Accuracy vs Coverage Trade-off\n"
    "(each line = one model, points = certitude levels 0.0 → 0.7 → 1.0)",
    fontsize=12, color=TEXT, y=1.0)

for model, grp in df_best.groupby("model_name"):
    grp_s = grp.sort_values("certitude")
    color = grp_s["color"].iloc[0]
    ax2.plot(grp_s["coverage_pct"], grp_s["accuracy_pct"],
             color=color, lw=1.4, alpha=0.7, zorder=2)
    for _, row in grp_s.iterrows():
        sty = CERTITUDE_STYLES[row["certitude"]]
        ax2.scatter(row["coverage_pct"], row["accuracy_pct"],
                    color=color, marker=sty["marker"],
                    s=sty["ms"] ** 2, zorder=4)
    # Label the certitude=0 point (full data)
    pt = grp_s[grp_s["certitude"] == 0.0].iloc[0]
    ax2.annotate(short(model),
                 (pt["coverage_pct"], pt["accuracy_pct"]),
                 textcoords="offset points", xytext=(6, 2),
                 fontsize=7.5, color=color, alpha=0.9)

# Legend for certitude markers
for cert, sty in CERTITUDE_STYLES.items():
    ax2.scatter([], [], color=MUTED, marker=sty["marker"],
                s=sty["ms"] ** 2, label=sty["label"])
ax2.legend(fontsize=9, loc="lower left")

ax2.set_xlabel("Coverage %  (fraction of pairs evaluated)", color=TEXT)
ax2.set_ylabel("Accuracy %", color=TEXT)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# Colour legend
for name, c in MODEL_COLORS.items():
    ax2.scatter([], [], color=c, s=50, label=name.replace("_", " "))
ax2.legend(fontsize=8, loc="lower left", ncol=2)

plt.tight_layout()
fig2.savefig(os.path.join(args.out_dir, "fig2_accuracy_vs_coverage.png"),
             dpi=150, bbox_inches="tight")
print("[✓] fig2_accuracy_vs_coverage.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Certitude effect: accuracy gain per model
# ─────────────────────────────────────────────────────────────────────────────
# Compute accuracy gain from certitude=0 → 1.0
pivot_cert = df_best.pivot_table(
    index="model_name", columns="certitude", values="accuracy_pct")
pivot_cert["gain_07"] = pivot_cert[0.7] - pivot_cert[0.0]
pivot_cert["gain_10"] = pivot_cert[1.0] - pivot_cert[0.0]
pivot_cert["short"]   = pivot_cert.index.map(short)
pivot_cert = pivot_cert.sort_values("gain_10", ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.suptitle(
    "Accuracy Gain from Certitude Filtering\n"
    "(Δ vs certitude=0.0 baseline, best pooling per model)",
    fontsize=12, color=TEXT, y=1.0)

y = np.arange(len(pivot_cert))
h = 0.35
ax3.barh(y - h/2, pivot_cert["gain_07"], height=h,
         color="#38bdf8", label="certitude=0.7  (+Δ pp)", zorder=3)
ax3.barh(y + h/2, pivot_cert["gain_10"], height=h,
         color="#fbbf24", label="certitude=1.0  (+Δ pp)", zorder=3)

for i, (_, row) in enumerate(pivot_cert.iterrows()):
    ax3.text(row["gain_07"] + 0.05, i - h/2, f"+{row['gain_07']:.1f}",
             va="center", fontsize=8, color=TEXT)
    ax3.text(row["gain_10"] + 0.05, i + h/2, f"+{row['gain_10']:.1f}",
             va="center", fontsize=8, color=TEXT)

ax3.set_yticks(y)
ax3.set_yticklabels(pivot_cert["short"], fontsize=9)
ax3.set_xlabel("Accuracy gain (pp)", color=TEXT)
ax3.axvline(0, color=BORDER, lw=1)
ax3.legend(fontsize=9)

# Annotate coverage cost
ax3.text(0.98, 0.02,
         "Coverage at cert=0.7: ~82%\nCoverage at cert=1.0: ~37%",
         transform=ax3.transAxes, ha="right", va="bottom",
         fontsize=9, color=MUTED,
         bbox=dict(facecolor=SURFACE, edgecolor=BORDER, alpha=0.8, pad=6))

plt.tight_layout()
fig3.savefig(os.path.join(args.out_dir, "fig3_certitude_gain.png"),
             dpi=150, bbox_inches="tight")
print("[✓] fig3_certitude_gain.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Model-size scaling (Qwen3 only)
# ─────────────────────────────────────────────────────────────────────────────
SIZE_MAP = {
    "Qwen3-0.6B-Base": 0.6, "Qwen3-0.6B": 0.6,  "Qwen3-Embedding-0.6B": 0.6,
    "Qwen3-1.7B-Base": 1.7, "Qwen3-1.7B": 1.7,
    "Qwen3-4B-Base":   4.0, "Qwen3-4B":   4.0,  "Qwen3-Embedding-4B":   4.0,
    "Qwen3-8B-Base":   8.0, "Qwen3-8B":   8.0,  "Qwen3-Embedding-8B":   8.0,
}

df0_qwen = df0[df0["family"] == "qwen3"].copy()
df0_qwen["size"] = df0_qwen["short_name"].map(SIZE_MAP)
df0_qwen = df0_qwen.dropna(subset=["size"])

fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.suptitle(
    "Model-Size Scaling — Qwen3 Family  (certitude=0.0)",
    fontsize=12, color=TEXT)

groups = {
    "Base":      (df0_qwen[df0_qwen["variant"] == "base"],      MODEL_COLORS["qwen3_base"],      "o", "--"),
    "Final":     (df0_qwen[df0_qwen["variant"] == "final"],     MODEL_COLORS["qwen3_final"],     "s", "-"),
    "Embedding": (df0_qwen[df0_qwen["variant"] == "embedding"], MODEL_COLORS["qwen3_embedding"], "^", "-"),
}

for label, (grp, color, marker, ls) in groups.items():
    grp = grp.sort_values("size")
    ax4.plot(grp["size"], grp["accuracy_pct"],
             color=color, marker=marker, ls=ls, lw=2, ms=9,
             label=label, zorder=3)
    for _, row in grp.iterrows():
        ax4.annotate(f"{row['accuracy_pct']:.1f}%",
                     (row["size"], row["accuracy_pct"]),
                     textcoords="offset points", xytext=(4, 5),
                     fontsize=8.5, color=color)

ax4.set_xlabel("Model size (B parameters)", color=TEXT)
ax4.set_ylabel("Accuracy %", color=TEXT)
ax4.set_xticks([0.6, 1.7, 4, 8])
ax4.set_xticklabels(["0.6B", "1.7B", "4B", "8B"])
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax4.legend(fontsize=10)

plt.tight_layout()
fig4.savefig(os.path.join(args.out_dir, "fig4_size_scaling.png"),
             dpi=150, bbox_inches="tight")
print("[✓] fig4_size_scaling.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Base vs Fine-tuned (Qwen3 generative, all certitude levels)
# ─────────────────────────────────────────────────────────────────────────────
sizes = [0.6, 1.7, 4.0, 8.0]
pairs = [
    ("Qwen/Qwen3-0.6B-Base",  "Qwen/Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B-Base",  "Qwen/Qwen3-1.7B"),
    ("Qwen/Qwen3-4B-Base",    "Qwen/Qwen3-4B"),
    ("Qwen/Qwen3-8B-Base",    "Qwen/Qwen3-8B"),
]

fig5, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig5.suptitle(
    "Base vs Fine-tuned (Qwen3 Generative) — Best Pooling",
    fontsize=12, color=TEXT, y=1.02)

for ax, cert in zip(axes, [0.0, 0.7, 1.0]):
    base_vals, final_vals, x_labels = [], [], []
    for base_m, final_m in pairs:
        b_row = df_best[(df_best["model_name"] == base_m)  & (df_best["certitude"] == cert)]
        f_row = df_best[(df_best["model_name"] == final_m) & (df_best["certitude"] == cert)]
        if b_row.empty or f_row.empty:
            continue
        base_vals.append(b_row.iloc[0]["accuracy_pct"])
        final_vals.append(f_row.iloc[0]["accuracy_pct"])
        x_labels.append(short(base_m).replace("-Base", ""))

    x = np.arange(len(x_labels))
    h = 0.3
    ax.barh(x - h/2, base_vals,  height=h, color=MODEL_COLORS["qwen3_base"],
            label="base", zorder=3)
    ax.barh(x + h/2, final_vals, height=h, color=MODEL_COLORS["qwen3_final"],
            label="final", zorder=3)

    for i, (b, f) in enumerate(zip(base_vals, final_vals)):
        ax.text(max(b, f) + 0.1, i, f"Δ{f-b:+.1f}", va="center",
                fontsize=8, color=TEXT if f > b else "#f87171")

    ax.set_yticks(x)
    ax.set_yticklabels(x_labels, fontsize=9)
    ax.set_title(f"certitude={cert}", fontsize=10, color=TEXT, pad=8)
    ax.set_xlabel("Accuracy %", fontsize=9, color=TEXT)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)
    ax.set_xlim(55, max(base_vals + final_vals) + 5)

plt.tight_layout()
fig5.savefig(os.path.join(args.out_dir, "fig5_base_vs_finetuned.png"),
             dpi=150, bbox_inches="tight")
print("[✓] fig5_base_vs_finetuned.png")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n── Top models (certitude=0, best pooling) ──")
print(df0[["short_name", "pooling", "accuracy_pct", "coverage_pct",
           "variant", "family"]].head(10).to_string(index=False))
print(f"\nPlots saved to: {os.path.abspath(args.out_dir)}/")
