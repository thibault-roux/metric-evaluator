import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ============================================================
# Configuration
# ============================================================

# Set this to match the language variant you want to plot, same
# convention as the pooling-table script ("_en" suffix vs none).
ENGLISH = True
ADDER = "_en" if ENGLISH else ""

RESULTS_DIR = Path(".")
GLOB_PATTERN = f"summary_results_*{ADDER}.csv"

OUTPUT_DIR = Path("plots/semdist_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Name of the column in the CSVs that holds the score to plot.
VALUE_COL = "accuracy_pct"

# Double-column paper figure size (matches the BERTScore script)
FIG_WIDTH = 3.45
FIG_HEIGHT = 2.7

# ============================================================
# Load data (one CSV per layer, layer number parsed from filename)
# ============================================================

dfs = []

for file in RESULTS_DIR.glob(GLOB_PATTERN):

    match = re.search(
        rf"summary_results_(-?\d+){re.escape(ADDER)}\.csv",
        file.name,
    )

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

df = df[["model_name", "pooling", VALUE_COL, "layer"]].copy()

df["layer"] = pd.to_numeric(df["layer"])
df[VALUE_COL] = pd.to_numeric(df[VALUE_COL])

# ============================================================
# Renumber layers per model: 1 = first layer, N = last layer.
#
# Raw layer indices are typically negative-from-the-end (e.g.
# -35 .. 0) and different models have different numbers of
# layers, so the offset has to be computed per model rather than
# globally.
# ============================================================

df["layer_display"] = (
    df.groupby("model_name")["layer"]
    .transform(lambda s: s - s.min() + 1)
)

# ============================================================
# Remap pooling names (same convention as the pooling-table script)
# ============================================================

df["pooling"] = df["pooling"].replace({
    "last_token": "Last",
    "last4_mean": "4 Last",
    "mean": "Mean",
    "mean_no_last_token": "Mean*",
    "penultimate_token": "2nd Last",
    "pooling": "Pool",

    # Qwen embedding models
    "qwen3_eos": "Last",

    "weighted_mean": "Wgt.",
    "weighted_mean_no_last_token": "Wgt.*",
})

# ============================================================
# Model families (same convention as the BERTScore script)
# ============================================================


def get_family(model):
    model_lower = model.lower()

    if (
        "bert" in model_lower
        or "roberta" in model_lower
        or "mpnet" in model_lower
        or "minilm" in model_lower
    ):
        return "BERT-like"

    if "qwen" in model_lower:
        return "Qwen"

    if (
        "llama" in model_lower
        or "bloom" in model_lower
        or "deepseek" in model_lower
    ):
        return "Other causal LLM"

    return "Other"


df["family"] = df["model_name"].apply(get_family)

# ============================================================
# Filter out specific models
# ============================================================

MODELS_TO_EXCLUDE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",

    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-Embedding-4B",

    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
]

df = df[~df["model_name"].isin(MODELS_TO_EXCLUDE)].copy()


def short_name(model):

    replacements = {
        "bigscience/bloom-": "BLOOM-",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-": "DeepSeek-Qwen-",
        "FacebookAI/": "",
        "google-bert/": "",
        "meta-llama/": "",
        "Qwen/": "",
        "sentence-transformers/": "",
    }

    name = model
    for old, new in replacements.items():
        name = name.replace(old, new)

    return name


family_order = [
    "BERT-like",
    "Qwen",
    "Other causal LLM",
]

model_order = {

    "BERT-like": [
        "google-bert/bert-base-uncased",
        "google-bert/bert-large-uncased",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v1",
        "sentence-transformers/all-mpnet-base-v2",
    ],

    "Qwen": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
    ],

    "Other causal LLM": [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ],
}

COLORS_BERT = {
    "google-bert/bert-base-uncased": "#3776AB",
    "google-bert/bert-large-uncased": "#174A73",
    "FacebookAI/roberta-base": "#2A9D8F",
    "FacebookAI/roberta-large": "#17665C",
    "sentence-transformers/all-MiniLM-L6-v2": "#9B72CF",
    "sentence-transformers/all-MiniLM-L12-v2": "#67408C",
    "sentence-transformers/all-mpnet-base-v1": "#D45087",
    "sentence-transformers/all-mpnet-base-v2": "#8F2858",
}

COLORS_QWEN = {
    "Qwen/Qwen3-0.6B": "#56B4E9",
    "Qwen/Qwen3-1.7B": "#2389C9",
    "Qwen/Qwen3-4B": "#1261A0",
    "Qwen/Qwen3-8B": "#083B66",
    "Qwen/Qwen3-Embedding-0.6B": "#F6B656",
    "Qwen/Qwen3-Embedding-4B": "#E5852B",
    "Qwen/Qwen3-Embedding-8B": "#B84A1B",
}

COLORS_CAUSAL = {
    "meta-llama/Llama-3.2-1B": "#0072B2",
    "meta-llama/Llama-3.2-3B": "#003F6B",
    "bigscience/bloom-560m": "#4C9F70",
    "bigscience/bloom-1b1": "#32855A",
    "bigscience/bloom-3b": "#21633F",
    "bigscience/bloom-7b1": "#123F2A",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "#D55E5E",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "#9E2F2F",
}

# ============================================================
# Pooling -> marker mapping. Each point on a line now shows the
# pooling method that achieved the best score at that layer.
# Describe this mapping in the figure caption in the paper.
# ============================================================

POOLING_MARKERS = {
    "Last": "o",
    "2nd Last": "P",
    "4 Last": "X",
    "Mean": "s",
    "Mean*": "^",
    "Wgt.": "D",
    "Wgt.*": "v",
}

_fallback_markers = ["*", "h", "8", "p", "<", ">"]
_fallback_idx = 0

for pooling in df["pooling"].dropna().unique():
    if pooling not in POOLING_MARKERS:
        if _fallback_idx >= len(_fallback_markers):
            raise RuntimeError(
                f"More unknown pooling methods than available fallback "
                f"markers. New pooling method: {pooling}"
            )
        POOLING_MARKERS[pooling] = _fallback_markers[_fallback_idx]
        _fallback_idx += 1

# ============================================================
# Reduce to a single "best pooling" point per (model, layer)
#
# This is what keeps the figures readable: one line per model
# instead of one line per (model, pooling) combination. The
# pooling method that won at a given layer is still visible via
# the marker shape at that point.
# ============================================================

best = (
    df.sort_values(
        ["model_name", "layer_display", VALUE_COL],
        ascending=[True, True, False],
    )
    .groupby(["model_name", "layer_display"], as_index=False)
    .first()
)

# ============================================================
# Plot
# ============================================================

for family in family_order:

    family_df = best[best["family"] == family].copy()

    if family_df.empty:
        continue

    models = [
        model
        for model in model_order[family]
        if model in family_df["model_name"].unique()
    ]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    poolings_used = set()

    for model in models:

        if family == "BERT-like":
            color = COLORS_BERT[model]
        elif family == "Qwen":
            color = COLORS_QWEN[model]
        elif family == "Other causal LLM":
            color = COLORS_CAUSAL[model]
        else:
            color = None

        # Embedding-tuned Qwen models get a dashed line, matching
        # the convention used in the BERTScore figures.
        linestyle = "--" if (family == "Qwen" and "Embedding" in model) else "-"

        model_df = family_df[family_df["model_name"] == model].sort_values(
            "layer_display"
        )

        # Line only (no marker here, markers are added per-pooling below)
        ax.plot(
            model_df["layer_display"],
            model_df[VALUE_COL],
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            marker="None",
            label=short_name(model),
            zorder=2,
        )

        # Markers, split by whichever pooling won at each layer
        for pooling in model_df["pooling"].dropna().unique():

            poolings_used.add(pooling)

            pooling_df = model_df[model_df["pooling"] == pooling]

            ax.scatter(
                pooling_df["layer_display"],
                pooling_df[VALUE_COL],
                color=color,
                marker=POOLING_MARKERS[pooling],
                s=4,
                zorder=3,
            )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel("Layer", fontsize=8.5)
    ax.set_ylabel("Agreement (%)", fontsize=8.5)
    ax.set_ylim(81, 88.1)  # <-- Set y-axis limits
    ax.tick_params(axis="both", labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.22, linewidth=0.6)

    # --------------------------------------------------------
    # Legends
    #
    # Two separate legends: one maps color -> model, the other
    # maps marker shape -> the pooling method that was best at
    # that point. Keep both handles so the color legend doesn't
    # get overwritten by the second ax.legend() call.
    # --------------------------------------------------------

    handles, labels = ax.get_legend_handles_labels()

    model_legend = ax.legend(
        handles,
        labels,
        fontsize=6.5,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#B0B0B0",
        framealpha=0.82,
        handlelength=2.5,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.5,
        title="Model",
        title_fontsize=6.5,
    )
    ax.add_artist(model_legend)

    pooling_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker=POOLING_MARKERS[pooling],
            markersize=4,
            linestyle="",
        )
        for pooling in sorted(poolings_used)
    ]

    ax.legend(
        pooling_handles,
        sorted(poolings_used),
        fontsize=6.5,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#B0B0B0",
        framealpha=0.82,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.5,
        title="Best pooling",
        title_fontsize=6.5,
    )

    fig.tight_layout(pad=0.5)

    # --------------------------------------------------------
    # Save PNG + vector PDF
    # --------------------------------------------------------

    filename = family.lower().replace(" ", "_").replace("-", "") + ".png"
    output_file = OUTPUT_DIR / filename

    fig.savefig(output_file, dpi=400, bbox_inches="tight")

    pdf_file = OUTPUT_DIR / filename.replace(".png", ".pdf")
    fig.savefig(pdf_file, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Saved: {pdf_file}")

    # Caption text for the pooling -> marker mapping used in this
    # figure, so it can be copied into the LaTeX \caption{} for the
    # corresponding figure.
    caption_bits = ", ".join(
        f"{pooling} = {POOLING_MARKERS[pooling]}"
        for pooling in sorted(poolings_used)
    )
    print(
        f"  Caption note ({family}): each line shows, for every layer, "
        f"the best-performing pooling strategy; marker shape indicates "
        f"which pooling was best at that layer ({caption_bits}). "
        f"Layers are renumbered per model from 1 (first layer) to N "
        f"(last layer)."
    )