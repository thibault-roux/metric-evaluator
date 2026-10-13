import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================
# Configuration
# ============================================================


ANNOTATIONS_FILE = Path("datasets/hats-en.csv")

# METRIC_A_FILE = Path(
#     "./logs/inference_cache/bigscience_bloom-3b/layer_-8/mean__347f890f70dd.csv"
# )
# METRIC_A_FILE = Path(
#     "./logs/inference_cache/bigscience_bloom-3b/layer_-7/mean__076abac7b2ff.csv"
# )
# METRIC_A_FILE = Path(
#     "./logs/inference_cache/bigscience_bloom-3b/layer_-9/mean__153017ec9965.csv"
# )
METRIC_A_FILE = Path(
    "./logs/inference_cache/bigscience_bloom-3b/layer_-10/mean__116d55eb2829.csv"
)

# METRIC_A_FILE = Path(
#     "./logs/inference_cache/wer/layer_0/lexical.csv"
# )

# METRIC_B_FILE = Path(
#     "./logs/inference_cache/Qwen_Qwen3-Embedding-8B/layer_-2/qwen3_eos__609623f0b669.csv"
# )
# METRIC_B_FILE = Path(
#     "./logs/inference_cache/cer/layer_0/lexical.csv"
# )
# METRIC_B_FILE = Path(
#     "./logs/inference_cache/wer/layer_0/lexical.csv"
# )
METRIC_B_FILE = Path(
    "./logs/inference_bertscore_cache/Qwen_Qwen3-Embedding-8B__layers4.csv"
)
# METRIC_B_FILE = Path(
#     "./logs/inference_bertscore_cache/bigscience_bloom-3b__layers4.csv"
# )



# ============================================================
# Load annotations
# ============================================================

all_annotations = pd.read_csv(ANNOTATIONS_FILE)

# IMPORTANT:
# item_idx=0 corresponds to the FIRST DATA ROW in hats-en.csv.
#
# We assign this index BEFORE filtering non-unanimous examples.
all_annotations["item_idx"] = range(len(all_annotations))

print("=" * 70)
print("HUMAN ANNOTATIONS")
print("=" * 70)

print(f"Total annotation rows:       {len(all_annotations)}")


# ============================================================
# Keep only unanimous human judgments
# ============================================================

unanimous_only = True

if unanimous_only:

    unanimous = (
        ((all_annotations["h1_votes"] == 3) & (all_annotations["h2_votes"] == 0))
        |
        ((all_annotations["h1_votes"] == 0) & (all_annotations["h2_votes"] == 3))
    )

    annotations = all_annotations[unanimous].copy()

else:
    annotations = all_annotations.copy()

print(f"Unanimous rows used:         {len(annotations)}")
print(f"Ignored non-unanimous rows:  {len(all_annotations) - len(annotations)}")


# ============================================================
# Human choice
# ============================================================

# 1 = H1 preferred
# 2 = H2 preferred

annotations["human_choice"] = (
    annotations["h1_votes"] < annotations["h2_votes"]
).astype(int) + 1


# ============================================================
# Load metric files
# ============================================================

def load_metric(path):
    df = pd.read_csv(path)

    required_columns = {
        "item_idx",
        "sim_ref_hypA",
        "sim_ref_hypB",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return df


metric_a = load_metric(METRIC_A_FILE)
metric_b = load_metric(METRIC_B_FILE)


# ============================================================
# Sanity checks
# ============================================================

print()
print("=" * 70)
print("INDEX CHECK")
print("=" * 70)

print(
    f"Annotations: item_idx {annotations.item_idx.min()} "
    f"to {annotations.item_idx.max()}"
)

print(
    f"Metric A:    item_idx {metric_a.item_idx.min()} "
    f"to {metric_a.item_idx.max()}"
)

print(
    f"Metric B:    item_idx {metric_b.item_idx.min()} "
    f"to {metric_b.item_idx.max()}"
)


# Check that metric indices are unique
if metric_a["item_idx"].duplicated().any():
    raise ValueError("Metric A contains duplicate item_idx values.")

if metric_b["item_idx"].duplicated().any():
    raise ValueError("Metric B contains duplicate item_idx values.")


# ============================================================
# Match metrics to HUMAN annotations
# ============================================================

merged = (
    annotations[
        [
            "item_idx",
            "triplet_id",
            "h1_votes",
            "h2_votes",
            "human_choice",
        ]
    ]
    .merge(
        metric_a[
            [
                "item_idx",
                "sim_ref_hypA",
                "sim_ref_hypB",
            ]
        ].rename(
            columns={
                "sim_ref_hypA": "A_sim_h1",
                "sim_ref_hypB": "A_sim_h2",
            }
        ),
        on="item_idx",
        how="inner",
    )
    .merge(
        metric_b[
            [
                "item_idx",
                "sim_ref_hypA",
                "sim_ref_hypB",
            ]
        ].rename(
            columns={
                "sim_ref_hypA": "B_sim_h1",
                "sim_ref_hypB": "B_sim_h2",
            }
        ),
        on="item_idx",
        how="inner",
    )
)


print(f"Matched triplets:           {len(merged)}")


# ============================================================
# Check for missing metric predictions
# ============================================================

missing_a = set(annotations.item_idx) - set(metric_a.item_idx)
missing_b = set(annotations.item_idx) - set(metric_b.item_idx)

if missing_a:
    print(f"WARNING: {len(missing_a)} annotations missing Metric A")

if missing_b:
    print(f"WARNING: {len(missing_b)} annotations missing Metric B")


# ============================================================
# Metric decision
# ============================================================

def metric_choice(score_h1, score_h2):
    """
    1 = chooses H1
    2 = chooses H2
    0 = tie -> considered INCORRECT
    """

    if score_h1 > score_h2:
        return 1

    if score_h2 > score_h1:
        return 2

    return 0


merged["metric_a_choice"] = [
    metric_choice(h1, h2)
    for h1, h2 in zip(
        merged["A_sim_h1"],
        merged["A_sim_h2"],
    )
]

merged["metric_b_choice"] = [
    metric_choice(h1, h2)
    for h1, h2 in zip(
        merged["B_sim_h1"],
        merged["B_sim_h2"],
    )
]


# ============================================================
# Agreement with human perception
# ============================================================

merged["metric_a_correct"] = (
    merged["metric_a_choice"] == merged["human_choice"]
)

merged["metric_b_correct"] = (
    merged["metric_b_choice"] == merged["human_choice"]
)

n = len(merged)

a_correct = int(merged["metric_a_correct"].sum())
b_correct = int(merged["metric_b_correct"].sum())

a_agreement = a_correct / n
b_agreement = b_correct / n


# ============================================================
# Metric ties
# ============================================================

a_ties = int((merged["metric_a_choice"] == 0).sum())
b_ties = int((merged["metric_b_choice"] == 0).sum())


# ============================================================
# McNemar contingency table
# ============================================================

#                         Metric B
#
#                     Correct    Wrong
#
# Metric A Correct       a          b
# Metric A Wrong         c          d

a = int(
    (
        merged["metric_a_correct"]
        & merged["metric_b_correct"]
    ).sum()
)

b = int(
    (
        merged["metric_a_correct"]
        & ~merged["metric_b_correct"]
    ).sum()
)

c = int(
    (
        ~merged["metric_a_correct"]
        & merged["metric_b_correct"]
    ).sum()
)

d = int(
    (
        ~merged["metric_a_correct"]
        & ~merged["metric_b_correct"]
    ).sum()
)

table = [[a, b], [c, d]]


# ============================================================
# McNemar tests
# ============================================================

# Exact test
mcnemar_exact = mcnemar(
    table,
    exact=True,
    correction=False,
)

# Approximate chi-square test with continuity correction
mcnemar_chi2 = mcnemar(
    table,
    exact=False,
    correction=True,
)


# Exact binomial test on discordant pairs
discordant = b + c

if discordant > 0:
    binomial_result = binomtest(
        min(b, c),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )

    binomial_p = binomial_result.pvalue
else:
    binomial_p = 1.0


# ============================================================
# Print results
# ============================================================

print()
print("=" * 70)
print("METRIC COMPARISON")
print("=" * 70)

print()
print("Metric A")
print("-" * 70)
print(f"File:                         {METRIC_A_FILE}")
print(f"Agreement with humans:        {a_agreement:.2%}")
print(f"Correct:                      {a_correct}/{n}")
print(f"Ties (counted as errors):     {a_ties}")
print(f"Tie rate:                     {a_ties / n:.2%}")

print()
print("Metric B")
print("-" * 70)
print(f"File:                         {METRIC_B_FILE}")
print(f"Agreement with humans:        {b_agreement:.2%}")
print(f"Correct:                      {b_correct}/{n}")
print(f"Ties (counted as errors):     {b_ties}")
print(f"Tie rate:                     {b_ties / n:.2%}")


print()
print("=" * 70)
print("AGREEMENT DIFFERENCE")
print("=" * 70)

difference = b_agreement - a_agreement

print(f"Metric A agreement:           {a_agreement:.2%}")
print(f"Metric B agreement:           {b_agreement:.2%}")
print(f"Difference (B - A):           {difference:+.2%}")
print(
    f"Difference in percentage pts: "
    f"{difference * 100:+.2f}"
)


print()
print("=" * 70)
print("PAIRED CONTINGENCY TABLE")
print("=" * 70)

print()
print("                         Metric B")
print("                     Correct    Wrong")
print(
    f"Metric A Correct     {a:7d}    {b:5d}"
)
print(
    f"Metric A Wrong       {c:7d}    {d:5d}"
)

print()
print(f"A correct, B wrong:            {b}")
print(f"A wrong, B correct:            {c}")
print(f"Discordant pairs:              {b + c}")


print()
print("=" * 70)
print("SIGNIFICANCE TESTS")
print("=" * 70)

print()
print("McNemar exact test:")
print(f"  p-value:                      {mcnemar_exact.pvalue:.6g}")

print()
print("McNemar chi-square test:")
print(f"  statistic:                    {mcnemar_chi2.statistic:.6f}")
print(f"  p-value:                      {mcnemar_chi2.pvalue:.6g}")

print()
print("Exact binomial test:")
print(f"  p-value:                      {binomial_p:.6g}")


# ============================================================
# Conclusion
# ============================================================

alpha = 0.05

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if mcnemar_exact.pvalue < alpha:

    if difference > 0:
        print(
            f"Metric B is significantly better than Metric A "
            f"(McNemar exact p={mcnemar_exact.pvalue:.6g})."
        )
    else:
        print(
            f"Metric A is significantly better than Metric B "
            f"(McNemar exact p={mcnemar_exact.pvalue:.6g})."
        )

else:

    print(
        f"The difference is NOT statistically significant "
        f"(McNemar exact p={mcnemar_exact.pvalue:.6g})."
    )