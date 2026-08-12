import argparse
import csv
import gc
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ------------------------------------------------------------
# Built-in model registry
# ------------------------------------------------------------
# Default panel:
# - Qwen3 base + final + embedding (0.6B / 1.7B / 4B / 8B where available)
# - Llama 3.2 base (1B / 3B)
# - Mistral 7B v0.3
# - OLMo 2 7B
#
# You can override the model list with --models ...

"""
MODEL_REGISTRY = [
    # Qwen3 base + final + embedding
    {"name": "Qwen/Qwen3-0.6B-Base", "family": "qwen3", "variant": "base", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-0.6B", "family": "qwen3", "variant": "final", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-1.7B-Base", "family": "qwen3", "variant": "base", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-1.7B", "family": "qwen3", "variant": "final", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-4B-Base", "family": "qwen3", "variant": "base", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-4B", "family": "qwen3", "variant": "final", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-8B-Base", "family": "qwen3", "variant": "base", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-8B", "family": "qwen3", "variant": "final", "kind": "generic_decoder"},
    {"name": "Qwen/Qwen3-Embedding-0.6B", "family": "qwen3", "variant": "embedding", "kind": "qwen3_embedding"},
    {"name": "Qwen/Qwen3-Embedding-4B", "family": "qwen3", "variant": "embedding", "kind": "qwen3_embedding"},
    {"name": "Qwen/Qwen3-Embedding-8B", "family": "qwen3", "variant": "embedding", "kind": "qwen3_embedding"},
    # Other decoder-only families around / under 8B
    # {"name": "meta-llama/Llama-3.2-1B", "family": "llama3.2", "variant": "base", "kind": "generic_decoder"},
    # {"name": "meta-llama/Llama-3.2-3B", "family": "llama3.2", "variant": "base", "kind": "generic_decoder"},
    {"name": "mistralai/Mistral-7B-v0.3", "family": "mistral", "variant": "base", "kind": "generic_decoder"},
    {"name": "allenai/OLMo-2-1124-7B", "family": "olmo2", "variant": "base", "kind": "generic_decoder"},
    # Gemma family (<= 8B, text pipeline compatible)
    {"name": "google/gemma-2b", "family": "gemma", "variant": "base", "kind": "generic_decoder"},
    {"name": "google/gemma-7b", "family": "gemma", "variant": "base", "kind": "generic_decoder"},
    {"name": "google/gemma-2-2b", "family": "gemma2", "variant": "base", "kind": "generic_decoder"},
    {"name": "google/gemma-3-1b-pt", "family": "gemma3", "variant": "base", "kind": "generic_decoder"},
]
"""

"""
MODEL_REGISTRY = [
    # Gemma family
    {
        "name": "google/gemma-3-27b-it",
        "family": "gemma3",
        "variant": "instruct",
        "size": "27b",
        "kind": "generic_decoder"
    },
    {
        "name": "google/gemma-4-31B",
        "family": "gemma4",
        "variant": "base",
        "size": "31b",
        "kind": "generic_decoder"
    },

    # Qwen3 family
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "family": "qwen3",
        "variant": "instruct_moe",
        "size": "30b",
        "kind": "generic_decoder"
    },
]
"""

"""
MODEL_REGISTRY = [
    # Qwen3.5 family
    {
        "name": "Qwen/Qwen3.5-27B",
        "family": "qwen3.5",
        "variant": "base",
        "size": "27b",
        "kind": "generic_decoder"
    },
]
"""

# MODEL_REGISTRY = [
#     {
#         "name": "Qwen/Qwen3.5-35B-A3B",
#         "family": "qwen3.5",
#         "variant": "moe",
#         "size": "35b",
#         "kind": "generic_decoder"
#     },
# ]

MODEL_REGISTRY = [
    {
        "name": "dangvantuan/sentence-camembert-base",
        "family": "roberta",
        "variant": "moe",
        "size": "0.1b",
        "kind": "generic_encoder"
    },
    {
        "name": "dangvantuan/sentence-camembert-large",
        "family": "roberta",
        "variant": "moe",
        "size": "0.3b",
        "kind": "generic_encoder"
    },
    {
        "name": "almanach/camembert-large",
        "family": "roberta",
        "variant": "base",
        "size": "0.3b",
        "kind": "generic_encoder"
    },
    {
        "name": "almanach/camembert-base",
        "family": "roberta",
        "variant": "base",
        "size": "0.1b",
        "kind": "generic_encoder"
    },
    {
        "name": "Qwen/Qwen3-0.6B",
        "family": "qwen3",
        "variant": "base",
        "size": "0.6b",
        "kind": "generic_decoder"
    },
    {
        "name": "Qwen/Qwen3-1.7B",
        "family": "qwen3",
        "variant": "base",
        "size": "1.7b",
        "kind": "generic_decoder"
    },
    {
        "name": "Qwen/Qwen3-4B",
        "family": "qwen3",
        "variant": "base",
        "size": "4b",
        "kind": "generic_decoder"
    },
    {
        "name": "Qwen/Qwen3-8B",
        "family": "qwen3",
        "variant": "base",
        "size": "8b",
        "kind": "generic_decoder"
    },
    {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "family": "qwen3",
        "variant": "embedding",
        "size": "0.6b",
        "kind": "qwen3_embedding"
    },
    {
        "name": "Qwen/Qwen3-Embedding-4B",
        "family": "qwen3",
        "variant": "embedding",
        "size": "4b",
        "kind": "qwen3_embedding"
    },
    {
        "name": "Qwen/Qwen3-Embedding-8B",
        "family": "qwen3",
        "variant": "embedding",
        "size": "8b",
        "kind": "qwen3_embedding"
    },
]


# GENERIC_POOLINGS = ["last_token", "mean", "weighted_mean", "mean_no_last_token", "weighted_mean_no_last_token", "penultimate_token", "last4_mean"]
# QWEN3_EMBEDDING_POOLINGS = ["qwen3_eos", "mean", "weighted_mean", "mean_no_last_token", "weighted_mean_no_last_token", "penultimate_token", "last4_mean"]
GENERIC_POOLINGS = ["last_token", "mean", "weighted_mean", "mean_no_last_token", "weighted_mean_no_last_token"]
QWEN3_EMBEDDING_POOLINGS = ["qwen3_eos", "mean", "weighted_mean", "mean_no_last_token", "weighted_mean_no_last_token"]


@dataclass
class DatasetItem:
    idx: int
    reference: str
    hypA: str
    nbrA: int
    hypB: str
    nbrB: int


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_dataset(dataset_path: str) -> List[DatasetItem]:
    dataset: List[DatasetItem] = []
    with open(dataset_path, "r", encoding="utf8") as f:
        next(f)  # skip header
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(f"Invalid line {idx + 2} in {dataset_path}: expected 5 tab-separated columns.")
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


def infer_kind(model_name: str) -> str:
    return "qwen3_embedding" if "Qwen3-Embedding" in model_name else "generic_decoder"


def resolve_model_specs(model_names: Optional[Sequence[str]]) -> List[Dict]:
    registry = {spec["name"]: spec for spec in MODEL_REGISTRY}
    if not model_names:
        return [dict(spec) for spec in MODEL_REGISTRY]

    specs = []
    for name in model_names:
        if name in registry:
            specs.append(dict(registry[name]))
        else:
            specs.append(
                {
                    "name": name,
                    "family": name.split("/")[0] if "/" in name else "custom",
                    "variant": "custom",
                    "kind": infer_kind(name),
                }
            )
    return specs


def get_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def pick_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        try:
            device = model.device
            if device is not None and device.type != "meta":
                return device
        except Exception:
            pass

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hf_model(model_name: str, args):
    logging.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = args.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    if torch.cuda.is_available():
        load_kwargs["device_map"] = args.device_map
    dtype = get_torch_dtype(args.dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = "auto"

    try:
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        logging.info("Loaded with AutoModel: %s", model_name)
    except Exception as e_auto:
        logging.warning("AutoModel failed for %s: %s", model_name, e_auto)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        logging.info("Loaded with AutoModelForCausalLM: %s", model_name)

    model.eval()
    return tokenizer, model


def cleanup_model(tokenizer, model) -> None:
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chunked(iterable: Sequence, batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


# def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     lengths = attention_mask.sum(dim=1) - 1
#     lengths = torch.clamp(lengths, min=0)
#     batch_idx = torch.arange(hidden.size(0), device=hidden.device)
#     return hidden[batch_idx, lengths]

def last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    # Works for both left- and right-padded batches.
    has_token = attention_mask.bool().any(dim=1)
    reversed_last = attention_mask.size(1) - 1 - torch.argmax(attention_mask.flip(dims=[1]), dim=1)
    return torch.where(has_token, reversed_last, torch.zeros_like(reversed_last))

def exclude_last_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # Zero out the last non-padding token for each row.
    mask = attention_mask.clone()
    has_token = attention_mask.bool().any(dim=1)
    if has_token.any():
        batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)[has_token]
        last_idx = last_nonpad_indices(attention_mask)[has_token]
        mask[batch_idx, last_idx] = 0
    return mask


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = last_nonpad_indices(attention_mask)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]

def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def weighted_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Position weights 1,2,3,... over non-padding tokens only
    weights = attention_mask.cumsum(dim=1) * attention_mask
    weights = weights.to(hidden.dtype)
    weighted_hidden = hidden * weights.unsqueeze(-1)
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return weighted_hidden.sum(dim=1) / denom

def penultimate_token_pool(hidden, attention_mask):
    last_idx = last_nonpad_indices(attention_mask)
    penult_idx = torch.clamp(last_idx - 1, min=0)

    batch_idx = torch.arange(hidden.size(0), device=hidden.device)

    return hidden[batch_idx, penult_idx]

def last4_token_pool(hidden, attention_mask):
    outputs = []

    for h, m in zip(hidden, attention_mask):
        valid = h[m.bool()]
        outputs.append(valid[-4:].mean(dim=0))

    return torch.stack(outputs)

def pooled_vectors_from_outputs(outputs, num_layer: int, attention_mask: torch.Tensor, pooling_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model forward pass did not return hidden_states. Make sure output_hidden_states=True.")

    # last_hidden = getattr(outputs, "last_hidden_state", None)
    # if last_hidden is None:
    #     last_hidden = hidden_states[num_layer]
    last_hidden = hidden_states[num_layer]

    results: Dict[str, torch.Tensor] = {}

    for pooling in pooling_names:
        if pooling in ("last_token", "qwen3_eos"):
            pooled = last_token_pool(last_hidden, attention_mask)
        elif pooling == "mean":
            pooled = mean_pool(last_hidden, attention_mask)
        elif pooling == "weighted_mean":
            pooled = weighted_mean_pool(last_hidden, attention_mask)
        elif pooling == "mean_no_last_token":
            pooled = mean_pool(last_hidden, exclude_last_token_mask(attention_mask))
        elif pooling == "weighted_mean_no_last_token":
            pooled = weighted_mean_pool(last_hidden, exclude_last_token_mask(attention_mask))
        elif pooling == "penultimate_token": # updated on 18th jun 2026
            # penult = hidden_states[-2] if len(hidden_states) >= 2 else last_hidden
            # pooled = mean_pool(penult, attention_mask)
            pooled = penultimate_token_pool(last_hidden, attention_mask)
        elif pooling == "last4_mean": # updated on 18th jun 2026
            # selected = hidden_states[-4:] if len(hidden_states) >= 4 else hidden_states
            # stacked = torch.stack([x.to(torch.float32) for x in selected], dim=0).mean(dim=0)
            # pooled = mean_pool(stacked, attention_mask)
            pooled = last4_token_pool(last_hidden, attention_mask)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        pooled = F.normalize(pooled.to(torch.float32), p=2, dim=1)
        results[pooling] = pooled.cpu()

    return results


def encode_texts(
    tokenizer,
    model,
    num_layer: int,
    texts: Sequence[str],
    pooling_names: Sequence[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    input_device = pick_input_device(model)
    batch = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch = {k: v.to(input_device) for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

    pooled = pooled_vectors_from_outputs(outputs, num_layer, batch["attention_mask"], pooling_names)
    del batch, outputs
    return pooled


def select_poolings(model_spec: Dict, generic_poolings: Sequence[str]) -> List[str]:
    if model_spec["kind"] == "qwen3_embedding":
        poolings = list(dict.fromkeys(QWEN3_EMBEDDING_POOLINGS + list(generic_poolings)))
        if "last_token" in poolings:
            poolings.remove("last_token")
        return poolings
    return list(generic_poolings)


def evaluate_scores(
    dataset: Sequence[DatasetItem],
    simA: Sequence[float],
    simB: Sequence[float],
    certitude: float,
    min_votes: int,
    tie_epsilon: float,
) -> Dict[str, float]:
    total_items = len(dataset)
    filtered_low_votes = 0
    filtered_low_certitude = 0
    filtered_human_ties = 0
    ignored_model_ties = 0
    accepted_human_items = 0
    correct = 0
    incorrect = 0

    for item, score_a, score_b in zip(dataset, simA, simB):
        total_votes = item.nbrA + item.nbrB
        if total_votes < min_votes:
            filtered_low_votes += 1
            continue
        if item.nbrA == item.nbrB:
            filtered_human_ties += 1
            continue

        certainty = max(item.nbrA, item.nbrB) / total_votes
        if certainty < certitude:
            filtered_low_certitude += 1
            continue

        accepted_human_items += 1

        delta = score_a - score_b
        if abs(delta) <= tie_epsilon:
            ignored_model_ties += 1
            continue

        pred = "A" if delta > 0 else "B"
        gold = "A" if item.nbrA > item.nbrB else "B"
        if pred == gold:
            correct += 1
        else:
            incorrect += 1

    evaluated_pairs = correct + incorrect
    ignored_total = filtered_low_votes + filtered_low_certitude + filtered_human_ties + ignored_model_ties
    accuracy_pct = (100.0 * correct / evaluated_pairs) if evaluated_pairs > 0 else 0.0
    coverage_pct = (100.0 * evaluated_pairs / total_items) if total_items > 0 else 0.0
    eval_rate_pct = (100.0 * evaluated_pairs / accepted_human_items) if accepted_human_items > 0 else 0.0
    ignored_pct = (100.0 * ignored_total / total_items) if total_items > 0 else 0.0

    return {
        "total_items": total_items,
        "filtered_low_votes": filtered_low_votes,
        "filtered_low_certitude": filtered_low_certitude,
        "filtered_human_ties": filtered_human_ties,
        "accepted_human_items": accepted_human_items,
        "ignored_model_ties": ignored_model_ties,
        "evaluated_pairs": evaluated_pairs,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_pct": accuracy_pct,
        "coverage_pct": coverage_pct,
        "eval_rate_pct": eval_rate_pct,
        "ignored_pct": ignored_pct,
    }


def maybe_init_item_writer(item_scores_csv: Optional[str]):
    if not item_scores_csv:
        return None, None
    ensure_parent_dir(item_scores_csv)
    f = open(item_scores_csv, "w", newline="", encoding="utf8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "item_idx",
            "model_name",
            "family",
            "variant",
            "kind",
            "pooling",
            "sim_ref_hypA",
            "sim_ref_hypB",
            "sim_delta",
            "nbrA",
            "nbrB",
            "total_votes",
            "human_winner",
            "reference",
            "hypA",
            "hypB",
        ],
    )
    writer.writeheader()
    return f, writer


def append_summary_rows(summary_csv: str, rows: List[Dict], write_header: bool) -> None:
    ensure_parent_dir(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_one_model(model_spec: Dict, dataset: Sequence[DatasetItem], args, item_writer=None) -> List[Dict]:
    tokenizer, model = load_hf_model(model_spec["name"], args)
    poolings = select_poolings(model_spec, args.generic_poolings)
    logging.info("Poolings for %s: %s", model_spec["name"], ", ".join(poolings))

    all_scores = {
        pooling: {"simA": [], "simB": []}
        for pooling in poolings
    }

    batches = list(chunked(dataset, args.batch_size))
    pbar = tqdm(batches, desc=safe_slug(model_spec["name"])[:50], leave=False)

    for batch_items in pbar:
        refs = [x.reference for x in batch_items]
        hyps_a = [x.hypA for x in batch_items]
        hyps_b = [x.hypB for x in batch_items]
        texts = refs + hyps_a + hyps_b

        pooled = encode_texts(
            tokenizer=tokenizer,
            model=model,
            num_layer=args.num_layer,
            texts=texts,
            pooling_names=poolings,
            max_length=args.max_length,
        )

        n = len(batch_items)
        for pooling in poolings:
            vecs = pooled[pooling]
            ref_vecs = vecs[:n]
            hypa_vecs = vecs[n : 2 * n]
            hypb_vecs = vecs[2 * n : 3 * n]

            sim_a = (ref_vecs * hypa_vecs).sum(dim=1).tolist()
            sim_b = (ref_vecs * hypb_vecs).sum(dim=1).tolist()

            all_scores[pooling]["simA"].extend(sim_a)
            all_scores[pooling]["simB"].extend(sim_b)

            if item_writer is not None:
                for item, s_a, s_b in zip(batch_items, sim_a, sim_b):
                    human_winner = "A" if item.nbrA > item.nbrB else ("B" if item.nbrB > item.nbrA else "TIE")
                    item_writer.writerow(
                        {
                            "item_idx": item.idx,
                            "model_name": model_spec["name"],
                            "family": model_spec["family"],
                            "variant": model_spec["variant"],
                            "kind": model_spec["kind"],
                            "pooling": pooling,
                            "sim_ref_hypA": s_a,
                            "sim_ref_hypB": s_b,
                            "sim_delta": s_a - s_b,
                            "nbrA": item.nbrA,
                            "nbrB": item.nbrB,
                            "total_votes": item.nbrA + item.nbrB,
                            "human_winner": human_winner,
                            "reference": item.reference,
                            "hypA": item.hypA,
                            "hypB": item.hypB,
                        }
                    )

        pbar.set_postfix({"batch_size": len(batch_items)})

    cleanup_model(tokenizer, model)

    summary_rows: List[Dict] = []
    for pooling in poolings:
        for certitude in args.certitudes:
            metrics = evaluate_scores(
                dataset=dataset,
                simA=all_scores[pooling]["simA"],
                simB=all_scores[pooling]["simB"],
                certitude=certitude,
                min_votes=args.min_votes,
                tie_epsilon=args.tie_epsilon,
            )
            row = {
                "model_name": model_spec["name"],
                "family": model_spec["family"],
                "variant": model_spec["variant"],
                "kind": model_spec["kind"],
                "pooling": pooling,
                "certitude": certitude,
                "min_votes": args.min_votes,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                **metrics,
            }
            summary_rows.append(row)

    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairwise ref/hyp similarity evaluation with decoder-only models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the TSV dataset.")
    parser.add_argument("--summary_csv", type=str, default="results/summary_results.csv")
    parser.add_argument("--item_scores_csv", type=str, default=None, help="Optional per-item similarity CSV.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit HF model ids. If omitted, uses built-in decoder-only model set.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--min_votes", type=int, default=5)
    parser.add_argument("--certitudes", nargs="+", type=float, default=[1.0, 0.7, 0.0])
    parser.add_argument("--generic_poolings", nargs="+", default=GENERIC_POOLINGS)
    parser.add_argument("--tie_epsilon", type=float, default=1e-12)
    parser.add_argument("--padding_side", choices=["left", "right"], default="right")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_layer", type=int, default=-1, help="Which layer to pool from (0-based, -1 for last).")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logging.info("Reading dataset from %s", args.dataset_path)
    dataset = read_dataset(args.dataset_path)
    logging.info("Loaded %d examples", len(dataset))

    model_specs = resolve_model_specs(args.models)
    logging.info("Running %d models", len(model_specs))
    for spec in model_specs:
        logging.info("  - %s (%s / %s / %s)", spec["name"], spec["family"], spec["variant"], spec["kind"])

    ensure_parent_dir(args.summary_csv)
    if os.path.exists(args.summary_csv):
        os.remove(args.summary_csv)

    item_file, item_writer = maybe_init_item_writer(args.item_scores_csv)

    wrote_header = False
    models_bar = tqdm(model_specs, desc="Models")

    try:
        for spec in models_bar:
            models_bar.set_postfix({"model": safe_slug(spec["name"])[:24]})
            try:
                rows = run_one_model(spec, dataset, args, item_writer=item_writer)
                append_summary_rows(args.summary_csv, rows, write_header=not wrote_header)
                wrote_header = True

                best = max(rows, key=lambda x: x["accuracy_pct"])
                logging.info(
                    "Done %s | best pooling=%s | certitude=%.2f | acc=%.2f%% | evaluated=%d",
                    spec["name"],
                    best["pooling"],
                    best["certitude"],
                    best["accuracy_pct"],
                    best["evaluated_pairs"],
                )
            except Exception as exc:
                logging.exception("Model failed: %s", spec["name"])
                if args.fail_fast:
                    raise
                error_row = {
                    "model_name": spec["name"],
                    "family": spec["family"],
                    "variant": spec["variant"],
                    "kind": spec["kind"],
                    "pooling": "ERROR",
                    "certitude": "",
                    "min_votes": args.min_votes,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "total_items": len(dataset),
                    "filtered_low_votes": "",
                    "filtered_low_certitude": "",
                    "filtered_human_ties": "",
                    "accepted_human_items": "",
                    "ignored_model_ties": "",
                    "evaluated_pairs": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy_pct": 0.0,
                    "coverage_pct": 0.0,
                    "eval_rate_pct": 0.0,
                    "ignored_pct": 0.0,
                    "error": str(exc),
                }
                append_summary_rows(args.summary_csv, [error_row], write_header=not wrote_header)
                wrote_header = True
    finally:
        if item_file is not None:
            item_file.close()

    logging.info("Finished. Summary CSV: %s", args.summary_csv)
    if args.item_scores_csv:
        logging.info("Per-item score CSV: %s", args.item_scores_csv)


if __name__ == "__main__":
    main()
