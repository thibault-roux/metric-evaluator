import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("filtered_results.csv")

# Rename columns for paper
# df = df.rename(columns={
#     "last4_mean": "Last-4 Avg",
#     "last_token": "Final Token",
#     "mean": "Mean",
#     "mean_no_last_token": "Mean (No Last)",
#     "penultimate_token": "Penultimate Token",
#     "weighted_mean": "W-Mean",
#     "weighted_mean_no_last_token": "W-Mean (No Last)"
# })

df["model_name"] = df["model_name"].str.split("/").str[-1]

df = df.rename(columns={
    "last4_mean": "Moy. 4 derniers",
    "last_token": "Dernier token",
    "mean": "Moyenne",
    "mean_no_last_token": "Moy. sans dernier",
    "penultimate_token": "Avant-dernier",
    "weighted_mean": "Moy. pondérée",
    "weighted_mean_no_last_token": "Moy. pond. (sans dernier)"
})



# Set index
df = df.set_index("model_name")

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    df,
    annot=True,
    fmt=".0f",
    cmap="mako", # try: "viridis", "mako", "rocket"
    linewidths=0.5,
    cbar_kws={"label": "Score"}
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.ylabel("")

plt.title("Model Performance Across Metrics")
plt.tight_layout()

# Save as vector PDF
plt.savefig("plots_pooling/fres.pdf", bbox_inches="tight")
plt.close()