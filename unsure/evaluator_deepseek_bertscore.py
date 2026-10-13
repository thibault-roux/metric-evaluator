import os
import csv
import gc
import progressbar
import torch

from bert_score import BERTScorer


CACHE_DIR = "logs/inference_bertscore_cache"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
NUM_LAYERS = 28

DATASET_NAME = "hats-en.txt"

# Your original code used cert_X = 1.
CERTITUDE = 1.0


# ============================================================
# Cache utilities
# ============================================================

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").replace("\\", "_")


def cache_path(model_name, num_layers):
    fname = f"{sanitize_model_name(model_name)}__layers{num_layers}.csv"
    return os.path.join(CACHE_DIR, fname)


def read_cache(model_name, num_layers):
    path = cache_path(model_name, num_layers)

    if not os.path.exists(path):
        return None

    sims = []

    with open(
        path,
        "r",
        encoding="utf8",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            sims.append(
                (
                    float(row["sim_ref_hypA"]),
                    float(row["sim_ref_hypB"])
                )
            )

    return sims


def write_cache(model_name, num_layers, sims):
    os.makedirs(CACHE_DIR, exist_ok=True)

    path = cache_path(model_name, num_layers)

    with open(
        path,
        "w",
        encoding="utf8",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "item_idx",
                "sim_ref_hypA",
                "sim_ref_hypB"
            ]
        )

        for idx, (simA, simB) in enumerate(sims):
            writer.writerow([idx, simA, simB])


# ============================================================
# Dataset
# ============================================================

def read_dataset(dataname):
    dataset = []

    with open(
        "datasets/" + dataname,
        "r",
        encoding="utf8"
    ) as file:

        next(file)

        for line in file:
            line = line.rstrip("\n").split("\t")

            dictionary = {
                "reference": line[0],
                "hypA": line[1],
                "nbrA": int(line[2]),
                "hypB": line[3],
                "nbrB": int(line[4]),
            }

            dataset.append(dictionary)

    return dataset


# ============================================================
# Evaluation
# ============================================================

def evaluator(sims, dataset, certitude=0.3, verbose=True):
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0

    if verbose:
        bar = progressbar.ProgressBar(
            max_value=len(dataset)
        )

    for i in range(len(dataset)):

        if verbose:
            bar.update(i)

        nbrA = dataset[i]["nbrA"]
        nbrB = dataset[i]["nbrB"]

        if nbrA + nbrB < 3:
            ignored += 1
            continue

        maximum = max(nbrA, nbrB)
        c = maximum / (nbrA + nbrB)

        if c >= certitude:
            accepted += 1

            simA, simB = sims[i]

            # Higher BERTScore = more similar.
            if (
                (simA > simB and nbrA > nbrB)
                or
                (simB > simA and nbrB > nbrA)
            ):
                correct += 1
            else:
                incorrect += 1

        else:
            ignored += 1

    print()

    if correct + incorrect == 0:
        print(
            "No items met the certitude threshold; "
            "nothing to score."
        )
        return None

    accuracy = (
        correct / (correct + incorrect) * 100
    )

    ignored_ratio = (
        ignored / (ignored + accepted) * 100
    )

    print("ratio correct:", accuracy)
    print("ratio ignored:", ignored_ratio)

    return accuracy


# ============================================================
# All-layer BERTScore
# ============================================================

def compute_all_layer_scores(
    scorer,
    dataset,
    batch_size=1,
    verbose=True
):
    """
    Compute BERTScore for ALL hidden layers in one model pass.

    Returns:
        F1_A:
            Tensor of shape [num_items, num_layers + 1]

        F1_B:
            Tensor of shape [num_items, num_layers + 1]

    For DeepSeek-R1-Distill-Qwen-7B:

        index 0  = embedding output
        index 1  = transformer layer 1
        ...
        index 28 = transformer layer 28
    """

    refs = [
        item["reference"]
        for item in dataset
    ]

    hypsA = [
        item["hypA"]
        for item in dataset
    ]

    hypsB = [
        item["hypB"]
        for item in dataset
    ]

    if verbose:
        print()
        print("=" * 70)
        print("Scoring hypA against reference")
        print("=" * 70)

    _, _, F1_A = scorer.score(
        hypsA,
        refs,
        batch_size=batch_size,
        verbose=verbose
    )

    if verbose:
        print()
        print("=" * 70)
        print("Scoring hypB against reference")
        print("=" * 70)

    _, _, F1_B = scorer.score(
        hypsB,
        refs,
        batch_size=batch_size,
        verbose=verbose
    )

    # all_layers=True gives:
    #
    # [num_items, num_layers + 1]
    #
    # because layer 0 is the embedding output.

    F1_A = F1_A.detach().cpu()
    F1_B = F1_B.detach().cpu()

    return F1_A, F1_B


# ============================================================
# Convert one layer into the cache format used by your
# original script.
# ============================================================

def get_layer_sims(
    F1_A,
    F1_B,
    layer_idx
):
    """
    Extract one layer from all-layer BERTScore output.

    layer_idx:
        0  -> embedding output
        1  -> transformer layer 1
        ...
        28 -> transformer layer 28
    """

    sims = []

    for i in range(F1_A.shape[0]):
        sims.append(
            (
                F1_A[i, layer_idx].item(),
                F1_B[i, layer_idx].item()
            )
        )

    return sims


# ============================================================
# Main
# ============================================================

def main():

    model_name = MODEL_NAME

    print()
    print("=" * 70)
    print(
        f"Evaluating {model_name} "
        f"with ALL layers"
    )
    print("=" * 70)

    dataset = read_dataset(DATASET_NAME)

    # --------------------------------------------------------
    # Check whether every layer is already cached.
    # --------------------------------------------------------

    missing_layers = []

    for layer_idx in range(1, NUM_LAYERS + 1):

        cached = read_cache(
            model_name,
            layer_idx
        )

        if cached is None or len(cached) != len(dataset):
            missing_layers.append(layer_idx)

    if not missing_layers:
        print("All layer caches already exist.")

        for layer_idx in range(1, NUM_LAYERS + 1):

            sims = read_cache(
                model_name,
                layer_idx
            )

            x_score = evaluator(
                sims,
                dataset,
                certitude=CERTITUDE
            )

            if x_score is not None:
                os.makedirs("out", exist_ok=True)

                with open(
                    "out/results_bertscore_en.txt",
                    "a",
                    encoding="utf8"
                ) as file:

                    file.write(
                        f"{model_name}\t"
                        f"{layer_idx}\t"
                        f"{x_score:.2f}\n"
                    )

        return

    print(
        "Missing layers:",
        missing_layers
    )

    # --------------------------------------------------------
    # Load BERTScorer ONCE.
    #
    # all_layers=True means BERTScore asks the model for
    # hidden states from every layer in a single forward pass.
    # --------------------------------------------------------

    print()
    print("Loading BERTScorer once with all_layers=True...")

    scorer = BERTScorer(
        model_type=model_name,
        num_layers=NUM_LAYERS,
        all_layers=True,
        batch_size=1,
    )

    # DeepSeek/Qwen tokenizers sometimes have no pad token.
    if (
        model_name.startswith("meta-llama")
        and scorer._tokenizer.pad_token is None
    ):
        scorer._tokenizer.pad_token = (
            scorer._tokenizer.eos_token
        )

    scorer._tokenizer.model_max_length = 2048

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # BERTScorer has already moved the model to the GPU.
    #
    # At this point, if your bert-score installation loads
    # DeepSeek in FP32, the OOM can already have happened.
    #
    # Therefore the recommended solution is to modify
    # bert_score/utils.py as explained below.
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # Compute BOTH hypotheses while the same model is
        # resident in memory.
        # ----------------------------------------------------

        F1_A, F1_B = compute_all_layer_scores(
            scorer,
            dataset,
            batch_size=1,
            verbose=True
        )

        print()
        print(
            "All-layer score shapes:",
            F1_A.shape,
            F1_B.shape
        )

        # ----------------------------------------------------
        # DeepSeek has 28 transformer layers.
        #
        # all_layers=True gives:
        #
        #   index 0  = embedding output
        #   index 1  = transformer layer 1
        #   ...
        #   index 28 = transformer layer 28
        #
        # Your requested 28 transformer layers therefore use
        # indices 1..28.
        # ----------------------------------------------------

        for layer_idx in range(1, NUM_LAYERS + 1):

            print()
            print("=" * 70)
            print(
                f"Processing transformer layer "
                f"{layer_idx}/{NUM_LAYERS}"
            )
            print("=" * 70)

            # If the cache was already produced, don't overwrite it.
            cached = read_cache(
                model_name,
                layer_idx
            )

            if cached is not None and len(cached) == len(dataset):
                print(
                    f"Layer {layer_idx}: "
                    f"cache already exists, skipping."
                )
                continue

            sims = get_layer_sims(
                F1_A,
                F1_B,
                layer_idx
            )

            # Same cache format as your original script.
            write_cache(
                model_name,
                layer_idx,
                sims
            )

            print(
                f"Saved cache: "
                f"{cache_path(model_name, layer_idx)}"
            )

            # ------------------------------------------------
            # Evaluate this layer.
            # ------------------------------------------------

            x_score = evaluator(
                sims,
                dataset,
                certitude=CERTITUDE
            )

            if x_score is not None:

                os.makedirs(
                    "out",
                    exist_ok=True
                )

                with open(
                    "out/results_bertscore_en.txt",
                    "a",
                    encoding="utf8"
                ) as file:

                    file.write(
                        f"{model_name}\t"
                        f"{layer_idx}\t"
                        f"{x_score:.2f}\n"
                    )

                print(
                    f"Saved result: "
                    f"{model_name}\t"
                    f"{layer_idx}\t"
                    f"{x_score:.2f}"
                )

    finally:

        # Explicitly release GPU memory.
        del scorer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            print()
            print(
                "CUDA memory released."
            )


if __name__ == "__main__":
    main()
