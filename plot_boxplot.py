import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== CONFIG ====
# INPUT_CSV = "classified_with_semdist_2.csv"
# PLOT_FILE = "boxplot.png"
INPUT_CSV = input("Enter the path to the input CSV file (e.g., classified_with_semdist.csv): ")
PLOT_FILE = input("Enter the path to save the boxplot (e.g., boxplot.png): ")
# INPUT_CSV = "results/classified_output_gpt-4.1_with_semdist.csv"
# PLOT_FILE = "results/plot_gpt-4.1.pdf"


# if empty 
if INPUT_CSV == "":
    INPUT_CSV = "results/classified_output_qwen3.5-35b_with_semdist.csv"
if PLOT_FILE == "":
    PLOT_FILE = "results/plot_qwen3.5-35b_en.pdf"

lang = "en" # or "fr"

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# ==== BOX PLOT ====
if lang == "en":
    order = ["incomprehensible", "bad", "useful", "identical"]
elif lang == "fr":
    order = ["incompréhensible", "mauvaise", "utile", "identique"]
else:
    raise ValueError("Unsupported language. Use 'en' or 'fr'.")
colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
linecolors = ["#9c3227", "#9c5416", "#186d3b", "#194d71"]

if lang == "en":
    # replace in the class
    df["class"] = df["class"].replace({
        "incompréhensible": "incomprehensible",
        "mauvaise": "bad",
        "utile": "useful",
        "identique": "identical"
    })

plt.figure(figsize=(5, 4))
ax = sns.boxplot(
    x="class",
    y="sem_dist",
    data=df,
    order=order,
    palette=colors,
    width=0.2,
    flierprops=dict(marker="o", markersize=4, alpha=0.5)  # style des outliers
)

i = 0
for boxplot in ax.containers:
    # color = boxplot.boxes[0].get_facecolor()
    color = linecolors[i]
    i += 1
    plt.setp(boxplot.boxes, edgecolor=color)
    plt.setp(boxplot.caps, color=color)
    plt.setp(boxplot.fliers, color=color, markeredgecolor=color)
    plt.setp(boxplot.means, color=color)
    plt.setp(boxplot.medians, color=color)
    plt.setp(boxplot.whiskers, color=color)

# plt.title("Distribution des distances sémantiques par classe (boxplot)")
plt.ylabel("SemDist", fontweight="bold") # write in bold
if lang == "en":
    plt.xlabel("Predicted Class", fontweight="bold")
elif lang == "fr":
    plt.xlabel("Classe prédite", fontweight="bold")
else:
    raise ValueError("Unsupported language. Use 'en' or 'fr'.")
# rotate x labels and reduce font size
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# ==== SAVE ====
plt.savefig(PLOT_FILE, dpi=300)
print(f"Saved plot: {PLOT_FILE}")

plt.show()
