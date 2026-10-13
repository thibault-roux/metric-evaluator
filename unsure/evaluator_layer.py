import progressbar
from sentence_transformers import SentenceTransformer, util
import torch


def semdist_metric(ref, hyp, memory):
    model, layer_idx = memory
    
    # Encode with output_hidden_states
    ref_emb = encode_at_layer(model, ref, layer_idx)
    hyp_emb = encode_at_layer(model, hyp, layer_idx)
    
    similarity = util.cos_sim(ref_emb, hyp_emb).item()
    return 1 - similarity

def encode_at_layer(model, text, layer_idx):
    """Encode text and return embeddings from a specific transformer layer."""
    # Tokenize
    features = model.tokenize([text])
    features = {k: v.to(model.device) for k, v in features.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        output = model[0].auto_model(
            **features,
            output_hidden_states=True
        )
    
    # hidden_states: tuple of (num_layers+1) tensors of shape (batch, seq_len, hidden_size)
    hidden_states = output.hidden_states  # includes embedding layer at index 0
    
    # Get the chosen layer and mean-pool over tokens
    layer_output = hidden_states[layer_idx]  # (1, seq_len, hidden_size)
    attention_mask = features["attention_mask"]
    
    # Mean pooling (same as SentenceTransformer default)
    mask_expanded = attention_mask.unsqueeze(-1).float()
    pooled = (layer_output * mask_expanded).sum(1) / mask_expanded.sum(1)
    
    return pooled.squeeze(0)

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

def evaluator(metric, dataset, memory, certitude=0.3, verbose=True):
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
        
        if nbrA+nbrB < 5:
            ignored += 1
            continue
        maximum = max(nbrA, nbrB)
        c = maximum/(nbrA+nbrB)
        if c >= certitude: # if humans are certain about choice
            accepted += 1
            scoreA = metric(dataset[i]["reference"], dataset[i]["hypA"], memory=memory)
            scoreB = metric(dataset[i]["reference"], dataset[i]["hypB"], memory=memory)
            if (scoreA < scoreB and nbrA > nbrB) or (scoreB < scoreA and nbrB > nbrA):
                correct += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1
    print()
    print("ratio correct:", correct/(correct+incorrect)*100)
    print("ratio ignored:", ignored/(ignored+accepted)*100)
    return correct/(correct+incorrect)*100

if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1
    cert_Y = 0.7
    cert_Z = 0

    model_name = 'dangvantuan/sentence-camembert-large'
    # model_name = 'Lajavaness/sentence-camembert-large'
    # model_name = 'Qwen/Qwen3-Embedding-0.6B'
    # model_name = 'intfloat/multilingual-e5-large-instruct'
    # model_name = 'Qwen/Qwen3-Embedding-8B'
    # model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    # model_name = 'EuroBERT/EuroBERT-2.1B'
    # model_name = 'EuroBERT/EuroBERT-210m'
    # model_name = 'answerdotai/ModernBERT-base'
    # model_name = 'answerdotai/ModernBERT-large'
    # model_name = 'BAAI/bge-m3'

    model = SentenceTransformer(model_name)

    # Get total number of layers (including embedding layer at 0)
    num_layers = model[0].auto_model.config.num_hidden_layers  # e.g. 24 for large
    print(f"Model has {num_layers} transformer layers (+ embedding layer at index 0)")

    # Sweep all layers
    results = {}
    for layer_idx in range(0, num_layers + 1):  # 0 = embeddings, 1..N = transformer layers
        print(f"\n--- Layer {layer_idx} ---")
        memory = (model, layer_idx)
        x_score = evaluator(semdist_metric, dataset, memory=memory, certitude=cert_X)
        y_score = evaluator(semdist_metric, dataset, memory=memory, certitude=cert_Y)
        z_score = evaluator(semdist_metric, dataset, memory=memory, certitude=cert_Z)
        results[layer_idx] = (x_score, y_score, z_score)

        with open("out/results_layers.txt", "a", encoding="utf8") as file:
            file.write(f"{model_name}\tlayer={layer_idx}\t{x_score:.2f}\t{y_score:.2f}\t{z_score:.2f}\n")

    # Print summary
    print("\n=== Summary ===")
    for layer_idx, (x_score, y_score, z_score) in results.items():
        print(f"Layer {layer_idx:2d} | cert=1.0: {x_score:.2f}% | cert=0.7: {y_score:.2f}% | cert=0.0: {z_score:.2f}%")