from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIG
# ==========================================================

RESULTS_DIR = Path(".")
OUTPUT_DIR = Path("plots")

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================================
# LOAD ALL CSV FILES
# ==========================================================

dfs = []

for csv_file in RESULTS_DIR.glob("summary_results_*.csv"):

    match = re.search(r"summary_results_(-?\d+)\.csv", csv_file.name)

    if not match:
        continue

    layer = int(match.group(1))

    df = pd.read_csv(csv_file)
    df["layer"] = layer

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df[
    [
        "model_name",
        "pooling",
        "accuracy_pct",
        "layer",
    ]
]

print(f"Loaded {len(df)} rows")
print(f"Models: {df['model_name'].nunique()}")

# ==========================================================
# SORT LAYERS
# ==========================================================

layer_order = sorted(df["layer"].unique())

# ==========================================================
# 1. LINE PLOTS PER MODEL
# ==========================================================

print("Generating line plots...")

for model in sorted(df["model_name"].unique()):

    subset = df[df["model_name"] == model]

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=subset,
        x="layer",
        y="accuracy_pct",
        hue="pooling",
        marker="o",
    )

    plt.title(model)
    plt.xlabel("Layer")
    plt.ylabel("Correlation (%)")

    plt.gca().invert_xaxis()

    plt.tight_layout()

    safe_name = re.sub(r"[^\w\-]", "_", model)

    plt.savefig(
        OUTPUT_DIR / f"lineplot_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ==========================================================
# 2. HEATMAP PER MODEL
# ==========================================================

print("Generating heatmaps...")

for model in sorted(df["model_name"].unique()):

    subset = df[df["model_name"] == model]

    pivot = subset.pivot_table(
        index="pooling",
        columns="layer",
        values="accuracy_pct",
        aggfunc="mean",
    )

    pivot = pivot.reindex(columns=layer_order)

    plt.figure(figsize=(10, 4))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="viridis",
    )

    plt.title(model)

    plt.tight_layout()

    safe_name = re.sub(r"[^\w\-]", "_", model)

    plt.savefig(
        OUTPUT_DIR / f"heatmap_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ==========================================================
# 3. BEST CONFIGURATION PER MODEL
# ==========================================================

best = (
    df.sort_values("accuracy_pct", ascending=False)
      .groupby("model_name")
      .first()
      .reset_index()
      .sort_values("accuracy_pct", ascending=False)
)

plt.figure(figsize=(12, max(6, len(best) * 0.4)))

sns.barplot(
    data=best,
    y="model_name",
    x="accuracy_pct",
)

plt.xlabel("Best correlation (%)")
plt.ylabel("Model")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "best_models_ranking.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# 4. BEST POOLING COUNTS
# ==========================================================

counts = best["pooling"].value_counts()

plt.figure(figsize=(7, 4))

sns.barplot(
    x=counts.index,
    y=counts.values,
)

plt.xlabel("Pooling")
plt.ylabel("Wins")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "best_pooling_counts.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# 5. GLOBAL HEATMAP
# ==========================================================
#
# For each model and layer:
# keep the BEST pooling score.
#
# ==========================================================

best_layer = (
    df.groupby(
        [
            "model_name",
            "layer",
        ]
    )["accuracy_pct"]
    .max()
    .reset_index()
)

pivot = best_layer.pivot(
    index="model_name",
    columns="layer",
    values="accuracy_pct",
)

pivot = pivot.reindex(columns=layer_order)

plt.figure(
    figsize=(
        12,
        max(6, len(pivot) * 0.4),
    )
)

sns.heatmap(
    pivot,
    cmap="viridis",
    annot=False,
)

plt.title("Best pooling score per layer")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "global_model_layer_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# SAVE BEST CONFIG TABLE
# ==========================================================

best.to_csv(
    OUTPUT_DIR / "best_configurations.csv",
    index=False,
)

print(f"All plots saved in: {OUTPUT_DIR.resolve()}")
