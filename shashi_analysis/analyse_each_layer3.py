from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

english = True
if english:
    adder="_en"
else:
    adder=""

# ==========================================================
# CONFIG
# ==========================================================
RESULTS_DIR = Path(".")

# ==========================================================
# LOAD DATA
# ==========================================================
dfs = []
for file in RESULTS_DIR.glob(f"summary_results_*{adder}.csv"):
    match = re.search(rf"summary_results_(-?\d+){re.escape(adder)}\.csv", file.name)
    if not match:
        continue
    layer = int(match.group(1))
    print(f"Loading {file} for layer {layer}")
    tmp = pd.read_csv(file)
    tmp["layer"] = layer
    dfs.append(tmp)

if not dfs:
    raise RuntimeError("No CSV files found or loaded")

df = pd.concat(dfs, ignore_index=True)
df = df[["model_name", "pooling", "accuracy_pct", "layer"]]

# filter all-MiniLM-L6-v2
df = df[df["model_name"] != "sentence-transformers/all-MiniLM-L6-v2"]

# Clean model names
df["model_name"] = df["model_name"].apply(lambda x: x.split("/")[-1])

# Remap pooling names
df["pooling"] = df["pooling"].replace({
    "last_token":                  "Last",
    "last4_mean":                  "4 Last",
    "mean":                        "Mean",
    "mean_no_last_token":          "Mean*",
    "penultimate_token":           "2nd Last",
    "pooling":                     "Pool",
    "qwen3_eos":                   "Last",
    "weighted_mean":               "Wgt.",
    "weighted_mean_no_last_token": "Wgt.*",
})

# ==========================================================
# BEST SCORE PER (MODEL, LAYER)
# ==========================================================
best = (
    df.sort_values("accuracy_pct", ascending=False)
      .groupby(["model_name", "layer"])
      .first()
      .reset_index()
)

# Pivot accuracy and pooling
scores = best.pivot(index="model_name", columns="layer", values="accuracy_pct")
pooling_grid = best.pivot(index="model_name", columns="layer", values="pooling")

# ==========================================================
# MAP POOLING → INTEGER for coloring
# ==========================================================
pooling_methods = sorted(best["pooling"].unique())
pooling_to_int = {p: i for i, p in enumerate(pooling_methods)}

pooling_int = pooling_grid.applymap(lambda x: pooling_to_int.get(x, np.nan))

# ==========================================================
# COLOR PALETTE — one color per pooling method
# ==========================================================
# Distinct, accessible colors
base_colors = [
    '#8EC3EB',
    '#FBB07A',
    '#9DD6A1',
    '#E0B3D6',
    '#E69F00',
    '#4F9FD6',
    '#F97316',
    '#5CB85C',
    '#C97BB7',
    '#E69F00',
    '#56B4E9',
    '#009E73',
    '#D55E00',
    '#E57F36',
    '#69B06A',
    '#BF90B3',
    '#609DC1',
    '#FD8D3C',
    '#74C476',
    '#D4A0C7',
    '#6BAED6',
    '#8C2D04',
    '#084594',
    '#005A32',
    '#7B2D8B',
]
pale_colors = [
    '#8EC3EB',
    '#FBB07A',
    '#9DD6A1',
    '#E0B3D6',
    '#E69F00',
]
dark_colors = [
    '#4F9FD6',
    '#F97316',
    '#5CB85C',
    '#C97BB7',
    "#E69D0077",
]
even_darker_colors = [
    '#2F6F9E',
    '#C85E12',
    '#3F8A3F',
    '#8F4F85',
    "#6C4A00",
]
colors = base_colors[: len(pooling_methods)]
cmap = ListedColormap(colors)

# ==========================================================
# PLOT
# ==========================================================
if english:
    weight = 1.3
else:
    weight = 1
y_axis = max(5, len(scores) * 0.35 * weight)
# fig, ax = plt.subplots(figsize=(5.5, max(5, len(scores) * 0.35)))
# update using y_axis for the figsize
fig, ax = plt.subplots(figsize=(5.5, y_axis))

n_rows, n_cols = pooling_int.shape
img_data = pooling_int.values.astype(float)

im = ax.imshow(
    img_data,
    cmap=cmap,
    aspect="auto",
    vmin=-0.5,
    vmax=len(pooling_methods) - 0.5,
)


# Grid lines
ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)


# After building `scores` and `pooling_grid`, compute best layer per model
best_layer = scores.idxmax(axis=1)  # Series: model → layer with highest accuracy

# Annotations: pooling name + accuracy
for i, model in enumerate(scores.index):
    for j, layer in enumerate(scores.columns):
        acc = scores.loc[model, layer]
        pool = pooling_grid.loc[model, layer]
        if pd.notna(acc) and pd.notna(pool):
            is_best = (layer == best_layer[model])
            ax.text(
                j, i,
                # f"{pool}\n{acc:.1f}",
                f"{acc:.1f}",
                ha="center", va="center",
                fontsize=8.5,
                # color="white",
                # color according to pooling method
                color=even_darker_colors[pooling_to_int[pool] % len(dark_colors)],
                fontweight="bold" if is_best else "normal",
            )



# Axes labels
ax.set_xticks(range(n_cols))
ax.set_xticklabels(scores.columns, fontsize=9)
ax.set_yticks(range(n_rows))
ax.set_yticklabels(scores.index, fontsize=9)
ax.set_xlabel("Layer", fontweight="bold", fontsize=11)
# ax.set_ylabel("Model", fontweight="bold", fontsize=11)



def wrap_model_name(name):
    if ("Embedding" not in name and "sentence" not in name) and "DeepSeek" not in name:
        return name
    parts = name.split("-")
    if "DeepSeek" in name:
        first_part = "-".join(parts[:2])
        next_part = "-".join(parts[2:])
        return f"{first_part}\n{next_part}"
    if len(parts) >= 2 and "DeepSeek" not in name:
        next_part = "-".join(parts[1:])
        return f"{parts[0]}\n{next_part}"
    return name

wrapped_labels = [wrap_model_name(m) for m in scores.index]
print(wrapped_labels)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(wrapped_labels, fontsize=9)

for label in ax.get_yticklabels():
    label.set_horizontalalignment('center')
    label.set_multialignment('center')


if english:
    ncol = 5
else:
    ncol = 4

# Legend
legend_handles = [
    mpatches.Patch(color=colors[i], label=p)
    for i, p in enumerate(pooling_methods)
]
ax.legend(
    handles=legend_handles,
    title="Pooling",
    title_fontsize=9,
    fontsize=8.5,
    ncol=ncol,              # one row with 4 entries
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),  # place above the axes
    borderaxespad=0,
    frameon=True,
)

ax.tick_params(axis='y', pad=40)

plt.tight_layout()
plt.savefig(f"summary_pooling_plot3{adder}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.savefig(f"summary_pooling_plot3{adder}.png", dpi=150, bbox_inches="tight", pad_inches=0.0)
plt.show()
print(f"Saved: summary_pooling_plot3{adder}.pdf / .png")