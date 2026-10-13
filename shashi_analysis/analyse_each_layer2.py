from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

RESULTS_DIR = Path(".")  # <-- adapte ça

# ==========================================================
# LOAD DATA
# ==========================================================

dfs = []

for file in RESULTS_DIR.glob("summary_results_*.csv"):

    match = re.search(r"summary_results_(-?\d+)\.csv", file.name)
    if not match:
        continue

    layer = int(match.group(1))

    tmp = pd.read_csv(file)
    tmp["layer"] = layer

    dfs.append(tmp)

if not dfs:
    raise RuntimeError("No CSV files found or loaded")

df = pd.concat(dfs, ignore_index=True)

# keep only useful columns
df = df[["model_name", "pooling", "accuracy_pct", "layer"]]

# ==========================================================
# BEST SCORE PER (MODEL, LAYER)
# ==========================================================

# all model name are like this: "Qwen/Qwen3-4B-Base"
# remove the first part of the model name (before the "/") to keep only "Qwen3-4B-Base"
df["model_name"] = df["model_name"].apply(lambda x: x.split("/")[-1])

# last_token: Last and qwen3_eos
# last4_mean: 4 Last
# mean: Mean
# mean_no_last_token: Mean*
# penultimate_token: 2nd Last
# weighted_mean: Wgt.
# weighted_mean_no_last_token: Wgt.*

# convert
df["pooling"] = df["pooling"].replace({
    "last_token": "Last",
    "last4_mean": "4 Last",
    "mean": "Mean",
    "mean_no_last_token": "Mean*",
    "penultimate_token": "2nd Last",
    "pooling": "Pool",
    "qwen3_eos": "Last",
    "weighted_mean": "Wgt.",
    "weighted_mean_no_last_token": "Wgt.*"
})


best = (
    df.sort_values("accuracy_pct", ascending=False)
      .groupby(["model_name", "layer"])
      .first()
      .reset_index()
)

scores = best.pivot(
    index="model_name",
    columns="layer",
    values="accuracy_pct"
)

# annotations: pooling + score
labels = (
    best["pooling"] # .str[0].str.upper()
    + "\n"
    + best["accuracy_pct"].round(1).astype(str)
)

labels = pd.DataFrame({
    "model_name": best["model_name"],
    "layer": best["layer"],
    "label": labels
}).pivot(
    index="model_name",
    columns="layer",
    values="label"
)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(14, max(6, len(scores) * 0.5)))

sns.heatmap(
    scores,
    cmap="viridis",
    annot=labels,
    fmt="",
    linewidths=0.5
)

# plt.title("Best accuracy per model and layer (with pooling)")
# put x and y labels in bold
plt.xlabel("Layer", fontweight="bold")
plt.ylabel("Model", fontweight="bold")

plt.tight_layout()
plt.savefig("summary_heatmap.pdf", dpi=300)
plt.show()
