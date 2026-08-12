import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns

# ==== CONFIG ====
# CSV_FILE = "classified_output.csv"
CSV_FILE = input("Enter the path to the CSV file (e.g., classified_output.csv): ")
# OUTPUT_CSV = "classified_with_semdist.csv"
# add with semdist to the csv file name
OUTPUT_CSV = CSV_FILE.replace(".csv", "_with_semdist2.csv")
# PLOT_FILE = "violin_plot_2.png"
PLOT_FILE = input("Enter the path to save the plot (e.g., violin_plot.png): ")
# SEM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEM_MODEL_NAME = "dangvantuan/sentence-camembert-large"
# SEM_MODEL_NAME = "Lajavaness/sentence-camembert-large"

# ==== LOAD DATA ====
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows from {CSV_FILE}")

# ==== LOAD MODEL ====
print("Loading semantic model...")
sem_model = SentenceTransformer(SEM_MODEL_NAME)

# ==== CALCUL SEMANTIC DISTANCE ====
def semdist_metric(ref, hyp, model):
    ref_emb = model.encode(ref, convert_to_tensor=True)
    hyp_emb = model.encode(hyp, convert_to_tensor=True)
    similarity = util.cos_sim(ref_emb, hyp_emb).item()
    return 1 - similarity

print("Computing semantic distances...")
df["sem_dist"] = df.apply(
    lambda row: semdist_metric(row["ref"], row["hyp"], sem_model),
    axis=1
)

# ==== SAVE CSV ====
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved CSV: {OUTPUT_CSV}")

# ==== VIOLIN PLOT ====
order = ["incompréhensible", "mauvaise", "utile", "identique"]


plt.figure(figsize=(10, 6))
sns.violinplot(
    x="class",
    y="sem_dist",
    data=df,
    order=order
)

plt.title("Distribution des distances sémantiques par classe")
plt.ylabel("Distance sémantique (1 - cosine similarity)")
plt.xlabel("Classe")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# ==== SAVE FIGURE ====
plt.savefig(PLOT_FILE, dpi=300)
print(f"Saved plot: {PLOT_FILE}")

# (optionnel, utile seulement en local avec interface graphique)
plt.show()
