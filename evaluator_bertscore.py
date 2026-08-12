import os
import csv
import progressbar
# sentence transformers
# from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer


CACHE_DIR = "logs/inference_bertscore_cache"


# def semdist_metric(ref, hyp, memory):
#     # example of a metric using sentence-transformers for semantic similarity
#     model = memory
#     ref_emb = model.encode(ref, convert_to_tensor=True)
#     hyp_emb = model.encode(hyp, convert_to_tensor=True)
#     similarity = util.cos_sim(ref_emb, hyp_emb).item()
#     return 1 - similarity


def sanitize_model_name(model_name):
    # turn "Qwen/Qwen3-0.6B" into "Qwen_Qwen3-0.6B" so it's a safe filename
    return model_name.replace("/", "_").replace("\\", "_")


def cache_path(model_name, num_layers):
    fname = f"{sanitize_model_name(model_name)}__layers{num_layers}.csv"
    return os.path.join(CACHE_DIR, fname)


def read_cache(model_name, num_layers):
    path = cache_path(model_name, num_layers)
    if not os.path.exists(path):
        return None
    sims = []
    with open(path, "r", encoding="utf8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sims.append((float(row["sim_ref_hypA"]), float(row["sim_ref_hypB"])))
    return sims


def write_cache(model_name, num_layers, sims):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(model_name, num_layers)
    with open(path, "w", encoding="utf8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["item_idx", "sim_ref_hypA", "sim_ref_hypB"])
        for idx, (simA, simB) in enumerate(sims):
            writer.writerow([idx, simA, simB])


def compute_sims(scorer, dataset, verbose=True):
    # batch-score all references/hypA and references/hypB at once for speed
    refs = [item["reference"] for item in dataset]
    hypsA = [item["hypA"] for item in dataset]
    hypsB = [item["hypB"] for item in dataset]

    if verbose:
        print("Scoring hypA against reference...")
    _, _, F1_A = scorer.score(hypsA, refs)

    if verbose:
        print("Scoring hypB against reference...")
    _, _, F1_B = scorer.score(hypsB, refs)

    sims = [(F1_A[i].item(), F1_B[i].item()) for i in range(len(dataset))]
    return sims


def get_or_compute_sims(scorer, dataset, model_name, num_layers, verbose=True):
    cached = read_cache(model_name, num_layers)
    if cached is not None and len(cached) == len(dataset):
        if verbose:
            print(f"Loaded cached scores from {cache_path(model_name, num_layers)}")
        return cached

    sims = compute_sims(scorer, dataset, verbose=verbose)
    write_cache(model_name, num_layers, sims)
    return sims


def read_dataset(dataname):
    # dataset = [{"reference": ref, "hypA": hypA, "nbrA": nbrA, "hypB": hypB, "nbrB": nbrB}, ...]
    dataset = []
    with open("datasets/" + dataname, "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dictionary = dict()
            dictionary["reference"] = line[0]
            dictionary["hypA"] = line[1]
            dictionary["nbrA"] = int(line[2])
            dictionary["hypB"] = line[3]
            dictionary["nbrB"] = int(line[4])
            dataset.append(dictionary)
    return dataset


def evaluator(sims, dataset, certitude=0.3, verbose=True):
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0

    if verbose:
        bar = progressbar.ProgressBar(max_value=len(dataset))
    for i in range(len(dataset)):
        if verbose:
            bar.update(i)
        nbrA = dataset[i]["nbrA"]
        nbrB = dataset[i]["nbrB"]

        if nbrA + nbrB < 5:
            ignored += 1
            continue
        maximum = max(nbrA, nbrB)
        c = maximum / (nbrA + nbrB)
        if c >= certitude:  # if humans are certain about choice
            accepted += 1
            simA, simB = sims[i]
            # metric is a distance (1 - F1), so lower F1 similarity = "farther"
            scoreA = 1 - simA
            scoreB = 1 - simB
            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                correct += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1
    print()
    print("ratio correct:", correct / (correct + incorrect) * 100)
    print("ratio ignored:", ignored / (ignored + accepted) * 100)
    return correct / (correct + incorrect) * 100


def main(model_name, num_layers=None):
    print("Reading dataset...")
    dataset = read_dataset("hats-en.txt")

    cert_X = 1
    # cert_Y = 0.7
    # cert_Z = 0

    # model = SentenceTransformer(model_name)
    model = BERTScorer(model_type=model_name, num_layers=num_layers)

    try:
        sims = get_or_compute_sims(model, dataset, model_name, num_layers)

        x_score = evaluator(sims, dataset, certitude=cert_X)  # certitude is useful to filter utterances where humans are unsure
        # y_score = evaluator(sims, dataset, certitude=cert_Y)
        # z_score = evaluator(sims, dataset, certitude=cert_Z)

        # with open("out/results_bertscore.txt", "a", encoding="utf8") as file:
        #     file.write(f"{model_name}\t{num_layers}\t{x_score:.2f}\t{y_score:.2f}\t{z_score:.2f}\n")

        with open("out/results_bertscore2.txt", "a", encoding="utf8") as file:
            file.write(f"{model_name}\t{num_layers}\t{x_score:.2f}\n")
    except AssertionError:
        print(f"Error evaluating {model_name} with num_layers={num_layers}. Skipping.")


if __name__ == '__main__':

    model2layer = {
        # ============================================================
        # Qwen 3
        # ============================================================
        "Qwen/Qwen3-0.6B": 28,
        
        # "Qwen/Qwen3-1.7B": 28,
        # "Qwen/Qwen3-4B": 36,
        # "Qwen/Qwen3-8B": 36,
        # "Qwen/Qwen3-Embedding-0.6B": 28,
        # "Qwen/Qwen3-Embedding-4B": 36,
        # "Qwen/Qwen3-Embedding-8B": 36,

        # # ============================================================
        # # BERT
        # # ============================================================
        # "google-bert/bert-base-uncased": 12,
        # "google-bert/bert-large-uncased": 24,

        # # ============================================================
        # # RoBERTa
        # # ============================================================
        # "FacebookAI/roberta-base": 12,
        # "FacebookAI/roberta-large": 24,

        # # ============================================================
        # # Llama 3.2
        # # ============================================================
        # "meta-llama/Llama-3.2-1B": 16,
        # "meta-llama/Llama-3.2-3B": 28,

        # # ============================================================
        # # Gemma 3
        # # ============================================================
        # "google/gemma-3-1b-it": 26,
        # "google/gemma-3-4b-it": 34,
        # "google/gemma-3-12b-it": 48,

        # # ============================================================
        # # DeepSeek R1 Distill
        # # ============================================================
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 28,
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 28,
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 48,

        # # ============================================================
        # # BLOOM
        # # ============================================================
        # "bigscience/bloom-560m": 24,
        # "bigscience/bloom-1b1": 24,
        # "bigscience/bloom-3b": 30,
        # "bigscience/bloom-7b1": 30,

        # # ============================================================
        # # Lightweight / fast
        # # ============================================================
        # "sentence-transformers/all-MiniLM-L6-v2": 6,
        # "sentence-transformers/all-MiniLM-L12-v2": 12,

        # # ============================================================
        # # MPNet
        # # ============================================================
        # "sentence-transformers/all-mpnet-base-v2": 12,
        # "sentence-transformers/all-mpnet-base-v1": 12,
    }

    models = []

    for model_name, num_layers in model2layer.items():
        for n_layers in range(num_layers):
            models.append((model_name, n_layers))

    for model_name, num_layers in models:
        print(f"Evaluating {model_name} with num_layers={num_layers}...")
        main(model_name, num_layers=num_layers)