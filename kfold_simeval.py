import argparse
import csv
import gc
import logging
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "dangvantuan/sentence-camembert-large"


@dataclass
class DatasetItem:
    idx: int
    reference: str
    hypA: str
    nbrA: int
    hypB: str
    nbrB: int


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_dataset(dataset_path: str) -> List[DatasetItem]:
    dataset = []

    with open(dataset_path, "r", encoding="utf8") as f:
        next(f)  # skip header

        for idx, line in enumerate(f):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 5:
                raise ValueError(
                    f"Invalid line {idx + 2}: expected 5 tab-separated columns."
                )

            dataset.append(
                DatasetItem(
                    idx=idx,
                    reference=parts[0],
                    hypA=parts[1],
                    nbrA=int(parts[2]),
                    hypB=parts[3],
                    nbrB=int(parts[4]),
                )
            )

    return dataset


def last_token_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Take the last non-padding token from each sequence.
    Works with both left and right padding.
    """

    # Index of last non-padding token.
    last_indices = attention_mask.sum(dim=1) - 1
    last_indices = torch.clamp(last_indices, min=0)

    batch_indices = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )

    return hidden[batch_indices, last_indices]


def encode_texts(
    tokenizer,
    model,
    texts,
    batch_size,
    max_length,
    device,
):
    all_embeddings = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding",
        leave=False,
    ):
        batch_texts = texts[start : start + batch_size]

        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        with torch.inference_mode():
            outputs = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last layer.
        hidden = outputs.hidden_states[-1]

        # Last non-padding token.
        embeddings = last_token_pool(
            hidden,
            batch["attention_mask"],
        )

        # Normalize so dot product = cosine similarity.
        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=1,
        )

        all_embeddings.append(embeddings.cpu())

        del batch
        del outputs
        del hidden
        del embeddings

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/kfold",
        help="Folder where the scores will be saved.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )

    args = parser.parse_args()

    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Reading dataset: %s", args.dataset_path)

    dataset = read_dataset(args.dataset_path)

    logging.info("Loaded %d examples", len(dataset))

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logging.info("Using device: %s", device)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    logging.info("Loading %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        # use_fast=True,
    )

    model_kwargs = {}

    if args.dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif args.dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Prepare texts
    # ------------------------------------------------------------

    references = [item.reference for item in dataset]
    hypotheses_a = [item.hypA for item in dataset]
    hypotheses_b = [item.hypB for item in dataset]

    # ------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------

    logging.info("Encoding references...")

    ref_embeddings = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=references,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    logging.info("Encoding hypothesis A...")

    hypA_embeddings = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=hypotheses_a,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    logging.info("Encoding hypothesis B...")

    hypB_embeddings = encode_texts(
        tokenizer=tokenizer,
        model=model,
        texts=hypotheses_b,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # ------------------------------------------------------------
    # Cosine similarities
    # ------------------------------------------------------------

    simA = (ref_embeddings * hypA_embeddings).sum(dim=1)
    simB = (ref_embeddings * hypB_embeddings).sum(dim=1)

    delta = simA - simB

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output_path = os.path.join(
        args.output_dir,
        "scores.csv",
    )

    logging.info("Saving scores to %s", output_path)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_idx",
                "sim_ref_hypA",
                "sim_ref_hypB",
                "sim_delta",
                "nbrA",
                "nbrB",
                "human_winner",
            ],
        )

        writer.writeheader()

        for i, item in enumerate(dataset):

            if item.nbrA > item.nbrB:
                human_winner = "A"
            elif item.nbrB > item.nbrA:
                human_winner = "B"
            else:
                human_winner = "TIE"

            writer.writerow(
                {
                    "item_idx": item.idx,
                    "sim_ref_hypA": float(simA[i]),
                    "sim_ref_hypB": float(simB[i]),
                    "sim_delta": float(delta[i]),
                    "nbrA": item.nbrA,
                    "nbrB": item.nbrB,
                    "human_winner": human_winner,
                }
            )

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata_path = os.path.join(
        args.output_dir,
        "metadata.txt",
    )

    with open(metadata_path, "w", encoding="utf8") as f:
        f.write(f"model={MODEL_NAME}\n")
        f.write("pooling=last_token\n")
        f.write("layer=last\n")
        f.write("similarity=cosine\n")
        f.write(f"max_length={args.max_length}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"num_items={len(dataset)}\n")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    del ref_embeddings
    del hypA_embeddings
    del hypB_embeddings
    del simA
    del simB
    del delta

    del tokenizer
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Done.")
    logging.info("Scores: %s", output_path)


if __name__ == "__main__":
    main()
