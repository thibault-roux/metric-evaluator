import pandas as pd
from pathlib import Path
from jiwer import wer, cer


# ============================================================
# Configuration
# ============================================================

ANNOTATIONS_FILE = Path("datasets/hats-en.csv")

OUTPUT_BASE = Path("./logs/inference_cache")


# ============================================================
# Load annotations
# ============================================================

df = pd.read_csv(ANNOTATIONS_FILE)

required_columns = {
    "reference",
    "h1_text",
    "h2_text",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"hats-en.csv is missing columns: {sorted(missing)}"
    )


# ============================================================
# Compute WER
# ============================================================

print("Computing WER...")

wer_h1 = []
wer_h2 = []

for _, row in df.iterrows():
    reference = str(row["reference"])
    h1 = str(row["h1_text"])
    h2 = str(row["h2_text"])

    wer_h1.append(1 - wer(reference, h1))
    wer_h2.append(1 - wer(reference, h2))


wer_output = pd.DataFrame({
    "item_idx": range(len(df)),
    "sim_ref_hypA": wer_h1,
    "sim_ref_hypB": wer_h2,
})


# ============================================================
# Compute CER
# ============================================================

print("Computing CER...")

cer_h1 = []
cer_h2 = []

for _, row in df.iterrows():
    reference = str(row["reference"])
    h1 = str(row["h1_text"])
    h2 = str(row["h2_text"])

    cer_h1.append(1 - cer(reference, h1))
    cer_h2.append(1 - cer(reference, h2))


cer_output = pd.DataFrame({
    "item_idx": range(len(df)),
    "sim_ref_hypA": cer_h1,
    "sim_ref_hypB": cer_h2,
})


# ============================================================
# Create directories
# ============================================================

wer_dir = OUTPUT_BASE / "wer" / "layer_0"
cer_dir = OUTPUT_BASE / "cer" / "layer_0"

wer_dir.mkdir(parents=True, exist_ok=True)
cer_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Save CSVs
# ============================================================

wer_path = wer_dir / "lexical.csv"
cer_path = cer_dir / "lexical.csv"

wer_output.to_csv(wer_path, index=False)
cer_output.to_csv(cer_path, index=False)


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(f"Number of triplets: {len(df)}")

print()
print(f"WER CSV: {wer_path}")
print(f"CER CSV: {cer_path}")

print()
print("WER:")
print(wer_output.head())

print()
print("CER:")
print(cer_output.head())