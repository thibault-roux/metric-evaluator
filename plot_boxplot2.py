import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np

# ==== CONFIG ====
inputs = [
    ("results/classified_output_gpt-4.1_with_semdist2.csv",    "GPT-4.1"),
    ("results/classified_output_qwen3.5-27b_with_semdist2.csv", "Qwen3.5-27B"),
    ("results/classified_output_qwen3.5-35b_with_semdist2.csv", "Qwen3.5-35B"),
    ("results/classified_output_gemma3-27b_with_semdist2.csv", "Gemma3-27B"),
    ("results/classified_output_gemma4-31b_with_semdist2.csv", "Gemma4-31B"),
    # ("results/classified_output_qwen3-0.6b_with_semdist2.csv", "Qwen3-0.6B"),
    ("results/classified_output_qwen3-1.7b_with_semdist2.csv", "Qwen3-1.7B"),
    ("results/classified_output_qwen3-30b_with_semdist2.csv", "Qwen3-30B")
]

PLOT_FILE = "results/plot_multi_llm"
lang = "en"  # "en" or "fr"

# ==== DESIGN TOKENS ====
# BACKGROUND  = "#F7F8FC"   # near-white blue-grey canvas
# PANEL_BG    = "#FFFFFF"   # pure white plot area
BACKGROUND = "#FFFFFF"
PANEL_BG = "#FFFFFF"
GRID_COLOR  = "#DDE3F0"   # soft slate grid
BORDER_COLOR = "#C8D0E5"  # muted frame border
TEXT_MAIN   = "#1A1F36"   # near-black ink
TEXT_SUB    = "#6B7499"   # muted sub-label

# Per-model palette: vivid but harmonious
MODEL_COLORS = [
    "#4361EE",  # electric blue  – model 1
    "#F72585",  # hot magenta    – model 2
    "#3ECFC4",  # teal           – model 3
    "#FF9F1C",  # amber          – model 4
    "#7B2FBE",  # violet         – model 5
    "#FF6D00",  # orange         – model 6
    "#2A9D8F",  # deep teal      – model 7
]

DISPLAY_FONT = "Poppins"
BODY_FONT    = "Lora"

# ==== CATEGORY ORDER & TRANSLATION ====
if lang == "en":
    order = ["incomprehensible", "bad", "useful", "identical"]
elif lang == "fr":
    order = ["incompréhensible", "mauvaise", "utile", "identique"]
else:
    raise ValueError("Unsupported language. Use 'en' or 'fr'.")

# ==== LOAD & COMBINE DATA ====
dfs = []
for file_path, model_name in inputs:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path} ({model_name})")
    if lang == "en":
        df["class"] = df["class"].replace({
            "incompréhensible": "incomprehensible",
            "mauvaise": "bad",
            "utile": "useful",
            "identique": "identical",
        })
    df["model"] = model_name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined  = combined[combined["class"].isin(order)]

model_names  = [label for _, label in inputs]
n_models     = len(model_names)
model_palette = {name: MODEL_COLORS[i % len(MODEL_COLORS)] for i, name in enumerate(model_names)}

# ==== FIGURE SETUP ====
fig_w = max(7, 1.8 * len(order) * n_models ** 0.6)
fig, ax = plt.subplots(figsize=(5, 4), facecolor=BACKGROUND)
ax.set_facecolor(PANEL_BG)

# Subtle top/right frame removal; keep left & bottom as thin lines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(BORDER_COLOR)
    ax.spines[spine].set_linewidth(0.8)

# ==== BOXPLOT ====
# Build alpha-tinted fill palette (60 % opacity rendered via RGBA hex)
def hex_to_rgba_fill(hex_color, alpha=0.18):
    """Convert #RRGGBB + alpha → (r,g,b,a) tuple for matplotlib."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r, g, b, alpha)

fill_palette = {name: hex_to_rgba_fill(MODEL_COLORS[i % len(MODEL_COLORS)], 0.22)
                for i, name in enumerate(model_names)}

bp = sns.boxplot(
    x="class",
    y="sem_dist",
    hue="model",
    data=combined,
    order=order,
    hue_order=model_names,
    palette=fill_palette,
    width=0.90,
    gap=0.5,
    linewidth=1.4,
    flierprops=dict(marker="o", markersize=3.5, alpha=0.45, markeredgewidth=0),
    medianprops=dict(linewidth=0),   # we'll draw our own median marker
    ax=ax,
)

# Re-colour every box element to match its model colour
n_cats     = len(order)
containers = ax.containers   # seaborn ≥ 0.12 exposes BoxPlotContainer list

for m_idx, container in enumerate(containers):
    color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
    dark  = color  # edges same hue, full opacity
    plt.setp(container.boxes,     edgecolor=dark, linewidth=1.4)
    plt.setp(container.whiskers,  color=dark,     linewidth=1.2, linestyle=(0,(4,2)))
    plt.setp(container.caps,      color=dark,     linewidth=1.2)
    plt.setp(container.fliers,    color=dark,     markeredgecolor=dark)
    plt.setp(container.medians,   color=dark,     linewidth=0)   # hide default line

# Draw a bold median tick on top of every box
for m_idx, container in enumerate(containers):
    color = MODEL_COLORS[m_idx % len(MODEL_COLORS)]
    for median_line in container.medians:
        x_data = median_line.get_xdata()
        y_val  = median_line.get_ydata()[0]
        x_mid  = np.mean(x_data)
        ax.plot(x_mid, y_val, marker="_", markersize=10,
                markeredgewidth=2.2, color=color, zorder=5,
                path_effects=[pe.withStroke(linewidth=3.5, foreground=PANEL_BG)])

# ==== GRID ====
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", zorder=0)
ax.set_axisbelow(True)

# ==== CATEGORY SEPARATORS ====
for i in range(1, n_cats):
    ax.axvline(i - 0.5, color=BORDER_COLOR, linewidth=0.6, zorder=0)

# ==== LABELS ====
ax.set_ylabel("SemDist",
              fontfamily=DISPLAY_FONT, fontsize=10,
              fontweight="bold", color=TEXT_MAIN, labelpad=10)

x_label = "Predicted Class" if lang == "en" else "Classe prédite"
ax.set_xlabel(x_label,
              fontfamily=DISPLAY_FONT, fontsize=10,
              fontweight="bold", color=TEXT_MAIN, labelpad=10)

# Styled x-tick labels
cat_display = {
    "incomprehensible": "Incompre-\nhensible",
    "bad":              "Bad",
    "useful":           "Useful",
    "identical":        "Identical",
    "incompréhensible": "Incompré-\nhensible",
    "mauvaise":         "Mauvaise",
    "utile":            "Utile",
    "identique":        "Identique",
}
ax.set_xticks(range(n_cats))
ax.set_xticklabels([cat_display.get(c, c) for c in order],
                   fontfamily=BODY_FONT, fontsize=8.5, color=TEXT_MAIN)
ax.tick_params(axis="y", labelsize=8, colors=TEXT_SUB)
ax.tick_params(axis="x", length=0)          # hide x tick marks

# ==== CUSTOM LEGEND ====
legend_handles = [
    mpatches.Patch(facecolor=hex_to_rgba_fill(MODEL_COLORS[i], 0.3),
                   edgecolor=MODEL_COLORS[i], linewidth=1.4,
                   label=name)
    for i, name in enumerate(model_names)
]
legend = ax.legend(
    handles=legend_handles,
    title="Model",
    title_fontproperties={"family": DISPLAY_FONT, "size": 8.5, "weight": "bold"},
    prop={"family": BODY_FONT, "size": 8},
    frameon=True,
    framealpha=0.92,
    edgecolor=BORDER_COLOR,
    facecolor=PANEL_BG,
    loc="upper right",
    borderpad=0.9,
    labelspacing=0.55,
)
legend.get_title().set_color(TEXT_MAIN)
for text in legend.get_texts():
    text.set_color(TEXT_MAIN)

# Remove seaborn's auto legend
old_legend = ax.get_legend()
if old_legend and old_legend is not legend:
    old_legend.remove()

ax.set_xlim(-0.5, len(order) - 0.5)

# ==== TIGHT LAYOUT & PADDING ====
fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.18)


fig.tight_layout(pad=0.3)

# ==== SAVE ====
plt.savefig(PLOT_FILE + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=BACKGROUND)
plt.savefig(PLOT_FILE + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=BACKGROUND)
print(f"Saved plot: {PLOT_FILE}")
plt.show()