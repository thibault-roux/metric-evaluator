from pathlib import Path
import re

import numpy as np
import pandas as pd


english = True

if english:
    adder = "_en"
else:
    adder = ""


# ==========================================================
# CONFIG
# ==========================================================

RESULTS_DIR = Path(".")


# ==========================================================
# LOAD DATA
# ==========================================================

dfs = []

for file in RESULTS_DIR.glob(f"summary_results_*{adder}.csv"):

    match = re.search(
        rf"summary_results_(-?\d+){re.escape(adder)}\.csv",
        file.name
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

df = df[
    [
        "model_name",
        "pooling",
        "accuracy_pct",
        "layer",
    ]
]


# ==========================================================
# FILTER MODEL
# ==========================================================

df = df[
    df["model_name"]
    != "sentence-transformers/all-MiniLM-L6-v2"
].copy()


# ==========================================================
# CLEAN MODEL NAMES
# ==========================================================

df["model_name"] = df["model_name"].apply(
    lambda x: x.split("/")[-1]
)


# Diagnostic: flag any model name containing a character outside
# the "safe" set. If your compile fails with an unclosed \begin{table*},
# check this list first -- an unescaped special character is the
# most common cause.

_safe_chars = re.compile(r"^[A-Za-z0-9\-\. ]+$")

for name in df["model_name"].unique():
    if not _safe_chars.match(name):
        print(f"WARNING: model name has unusual characters: {name!r}")


# ==========================================================
# REMAP POOLING NAMES
# ==========================================================

df["pooling"] = df["pooling"].replace({

    "last_token":                    "Last",
    "last4_mean":                    "4 Last",
    "mean":                          "Mean",
    "mean_no_last_token":            "Mean*",
    "penultimate_token":             "2nd Last",
    "pooling":                       "Pool",

    # Qwen embedding models
    "qwen3_eos":                     "Last",

    "weighted_mean":                 "Wgt.",
    "weighted_mean_no_last_token":   "Wgt.*",
})


# ==========================================================
# POOLING COLORS
# ==========================================================
#
# Add new pooling methods here as needed.
#
# The rest of the script will automatically include them
# if they occur in the data.
#
# Each pooling method gets its own distinct bg/text color
# pair (previously "Wgt." incorrectly reused the "Mean"
# colors -- fixed below).
# ==========================================================

pooling_colors = {

    "Last": {
        "bg": "poolLast",
        "text": "poolLastDark",
    },

    "Mean": {
        "bg": "poolMean",
        "text": "poolMeanDark",
    },

    "Mean*": {
        "bg": "poolMeanStar",
        "text": "poolMeanStarDark",
    },

    "Wgt.": {
        "bg": "poolWeighted",
        "text": "poolWeightedDark",
    },

    "Wgt.*": {
        "bg": "poolWeightedStar",
        "text": "poolWeightedStarDark",
    },
}


# ==========================================================
# LATEX COLORS
# ==========================================================

pooling_rgb = {

    # Background colors
    "poolLast":         "220,238,249",   # blue
    "poolMean":         "222,240,224",   # green
    "poolMeanStar":     "242,225,239",   # pink/purple
    "poolWeighted":     "250,235,210",   # orange (was a near-duplicate of green)
    "poolWeightedStar": "215,240,242",   # teal (was a near-duplicate of pink)

    # Dark text colors
    "poolLastDark":         "47,111,158",   # blue
    "poolMeanDark":         "63,138,63",    # green
    "poolMeanStarDark":     "143,79,133",   # pink/purple
    "poolWeightedDark":     "180,120,30",   # orange (was duplicating Mean's dark color)
    "poolWeightedStarDark": "35,128,130",   # teal (was duplicating Mean*'s dark color)
}


# ==========================================================
# AUTOMATICALLY HANDLE UNKNOWN POOLING METHODS
# ==========================================================

# Pooling methods actually present in the data
available_pooling = list(
    df["pooling"]
    .dropna()
    .unique()
)


# If a new pooling method appears in the CSV, don't crash.
# Give it a neutral color.
#
# You can replace these colors later with something prettier.

unknown_pooling_rgb = [
    ("poolExtraA", "238,238,238", "poolExtraADark", "80,80,80"),
    ("poolExtraB", "225,235,245", "poolExtraBDark", "60,90,120"),
    ("poolExtraC", "245,230,215", "poolExtraCDark", "130,90,50"),
    ("poolExtraD", "230,225,245", "poolExtraDDark", "90,70,130"),
]


extra_idx = 0

for pooling in available_pooling:

    if pooling in pooling_colors:
        continue

    if extra_idx >= len(unknown_pooling_rgb):
        raise RuntimeError(
            f"More unknown pooling methods than available "
            f"fallback colors. New pooling method: {pooling}"
        )

    bg, bg_rgb, text, text_rgb = unknown_pooling_rgb[extra_idx]

    pooling_colors[pooling] = {
        "bg": bg,
        "text": text,
    }

    pooling_rgb[bg] = bg_rgb
    pooling_rgb[text] = text_rgb

    extra_idx += 1


# ==========================================================
# BEST SCORE PER (MODEL, LAYER)
# ==========================================================

best = (
    df.sort_values(
        ["model_name", "layer", "accuracy_pct"],
        ascending=[True, True, False]
    )
    .groupby(["model_name", "layer"], as_index=False)
    .first()
)


# ==========================================================
# PIVOT
# ==========================================================

scores = best.pivot(
    index="model_name",
    columns="layer",
    values="accuracy_pct"
)

pooling_grid = best.pivot(
    index="model_name",
    columns="layer",
    values="pooling"
)


# Keep layers in numerical order

scores = scores.sort_index(axis=1)

pooling_grid = pooling_grid[scores.columns]


# ==========================================================
# BEST LAYER PER MODEL
# ==========================================================

best_layer = scores.idxmax(axis=1)


# ==========================================================
# MODEL GROUPING
# ==========================================================

def model_group(model):

    model_lower = model.lower()

    # ------------------------------------------------------
    # Encoder-based models
    # ------------------------------------------------------

    encoder_keywords = [
        "minilm",
        "mpnet",
        "bert-",
        "roberta-",
        "distilbert",
        "electra",
        "deberta",
    ]

    if any(keyword in model_lower for keyword in encoder_keywords):
        return "Encoder-based models"


    # ------------------------------------------------------
    # Qwen models
    # ------------------------------------------------------

    qwen_keywords = [
        "qwen",
        "deepseek-r1",
    ]

    if any(keyword in model_lower for keyword in qwen_keywords):
        return "Qwen models"


    # ------------------------------------------------------
    # Everything else
    # ------------------------------------------------------

    return "Other causal LLM"


# Add group column

model_groups = {
    model: model_group(model)
    for model in scores.index
}


# Desired group ordering

group_order = [
    "Encoder-based models",
    "Qwen models",
    "Other causal LLM",
]


# ==========================================================
# LATEX ESCAPING
# ==========================================================

def latex_escape(text):
    """
    Escape LaTeX special characters while preserving '-'.

    Backslash is escaped FIRST (as \\textbackslash{}) so that the
    backslashes introduced by the other replacements below don't
    get re-escaped.
    """

    text = str(text)

    # Backslash must be handled before anything else, otherwise the
    # backslashes we insert for &, %, $, etc. would themselves get
    # escaped on a later pass.
    text = text.replace("\\", r"\textbackslash{}")

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ==========================================================
# GENERATE LATEX
# ==========================================================

lines = []


# ----------------------------------------------------------
# Required packages
# ----------------------------------------------------------

lines.append(r"% Required packages:")
lines.append(r"% \usepackage{booktabs}")
lines.append(r"% \usepackage[table]{xcolor}")
lines.append(r"% \usepackage{graphicx}  % needed for \resizebox")
lines.append("")


# ----------------------------------------------------------
# Color definitions
# ----------------------------------------------------------

for color_name, rgb in pooling_rgb.items():

    lines.append(
        rf"\definecolor{{{color_name}}}{{RGB}}{{{rgb}}}"
    )

lines.append("")


# ==========================================================
# TABLE
# ==========================================================

n_layers = len(scores.columns)

column_spec = "l" + "c" * n_layers


# ----------------------------------------------------------
# Full-width table
# ----------------------------------------------------------

lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{4pt}")

# Wrap the tabular in \resizebox so the table never exceeds
# \textwidth, regardless of how many layer columns there are.
lines.append(r"\resizebox{\textwidth}{!}{%")

lines.append(
    rf"\begin{{tabular}}{{{column_spec}}}"
)

lines.append(r"\toprule")


# ==========================================================
# HEADER
# ==========================================================

header = ["Model"]

for layer in scores.columns:
    header.append(str(layer))

lines.append(
    " & ".join(header) + r" \\"
)

lines.append(r"\midrule")


# ==========================================================
# TABLE BODY
# ==========================================================

first_group = True

for group_name in group_order:

    # Models belonging to this group

    group_models = [
        model
        for model in scores.index
        if model_groups[model] == group_name
    ]

    if not group_models:
        continue


    # ------------------------------------------------------
    # Section header
    # ------------------------------------------------------

    if not first_group:
        lines.append(r"\midrule")

    first_group = False

    lines.append(
        rf"\multicolumn{{{n_layers + 1}}}{{l}}{{"
        rf"\textbf{{\textit{{{latex_escape(group_name)}}}}}"
        rf"}} \\"
    )

    lines.append(r"\midrule")


    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    for model in group_models:

        model_label = latex_escape(model)

        row = [model_label]

        for layer in scores.columns:

            acc = scores.loc[model, layer]

            pooling = pooling_grid.loc[model, layer]


            # --------------------------------------------------
            # Missing value
            # --------------------------------------------------

            if pd.isna(acc):

                row.append("--")
                continue


            # --------------------------------------------------
            # Best layer
            # --------------------------------------------------

            is_best = layer == best_layer[model]

            if is_best:

                accuracy_text = rf"\textbf{{{acc:.1f}}}"

            else:

                accuracy_text = f"{acc:.1f}"


            # --------------------------------------------------
            # Pooling color
            # --------------------------------------------------

            if pooling in pooling_colors:

                bg_color = pooling_colors[pooling]["bg"]
                text_color = pooling_colors[pooling]["text"]

                cell = (
                    rf"\cellcolor{{{bg_color}}}"
                    rf"{{\textcolor{{{text_color}}}"
                    rf"{{{accuracy_text}}}}}"
                )

            else:

                cell = accuracy_text


            row.append(cell)


        lines.append(
            " & ".join(row) + r" \\"
        )


    # ------------------------------------------------------
    # Space between sections
    # ------------------------------------------------------

    lines.append(r"\addlinespace")


# ==========================================================
# END TABULAR
# ==========================================================

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")  # close \resizebox


# ==========================================================
# CAPTION
# ==========================================================

# Only show pooling methods that actually occur.

legend_items = []

for pooling in available_pooling:

    if pooling not in pooling_colors:
        continue

    bg_color = pooling_colors[pooling]["bg"]

    legend_items.append(
        rf"\colorbox{{{bg_color}}}{{\strut {latex_escape(pooling)}}}"
    )


legend = ", ".join(legend_items)


lines.append(
    r"\caption{"
    r"Accuracy (\%) across layers. "
    r"Each cell reports the best accuracy obtained among the "
    r"available pooling strategies for the corresponding "
    r"model and layer. "
    r"Cell background colors indicate the selected pooling "
    r"strategy: "
    + legend
    + r". "
    r"The best layer for each model is shown in bold."
    r"}"
)


# ==========================================================
# LABEL
# ==========================================================

lines.append(
    r"\label{tab:summary_pooling}"
)

lines.append(
    r"\end{table*}"
)


# ==========================================================
# SAVE
# ==========================================================

output_file = (
    RESULTS_DIR
    / f"summary_pooling_table{adder}.tex"
)

with open(output_file, "w", encoding="utf-8") as f:

    f.write("\n".join(lines))


print(f"Saved: {output_file}")