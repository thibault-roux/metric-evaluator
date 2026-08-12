import progressbar
# sentence transformers
# from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer



# def semdist_metric(ref, hyp, memory):
#     # example of a metric using sentence-transformers for semantic similarity
#     model = memory
#     ref_emb = model.encode(ref, convert_to_tensor=True)
#     hyp_emb = model.encode(hyp, convert_to_tensor=True)
#     similarity = util.cos_sim(ref_emb, hyp_emb).item()
#     return 1 - similarity

def berscore_metric(ref, hyp, memory):
    # example of a metric using bertscore for semantic similarity
    scorer = memory
    P, R, F1 = scorer.score([hyp], [ref])
    return 1 - F1.item()

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

def main(model_name, num_layers=None):
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1
    # cert_Y = 0.7
    # cert_Z = 0

    # model = SentenceTransformer(model_name)
    model = BERTScorer(model_type=model_name, num_layers=num_layers)

    memory = model

    try:
        x_score = evaluator(berscore_metric, dataset, memory=memory, certitude=cert_X) # certitude is useful to filter utterances where humans are unsure    
        # y_score = evaluator(berscore_metric, dataset, memory=memory, certitude=cert_Y)
        # z_score = evaluator(berscore_metric, dataset, memory=memory, certitude=cert_Z)

        # with open("out/results_bertscore.txt", "a", encoding="utf8") as file:
        #     file.write(f"{model_name}\t{num_layers}\t{x_score:.2f}\t{y_score:.2f}\t{z_score:.2f}\n")

        with open("out/results_bertscore2.txt", "a", encoding="utf8") as file:
            file.write(f"{model_name}\t{num_layers}\t{x_score:.2f}\n")
    except AssertionError:
        print(f"Error evaluating {model_name} with num_layers={num_layers}. Skipping.")


if __name__ == '__main__':

    model2layer = { # number of layers in the models
        # "dangvantuan/sentence-camembert-base": 12,
        # "dangvantuan/sentence-camembert-large": 24,
        # "almanach/camembert-base": 12,
        # # "almanach/camembert-large": 24,
        # # "answerdotai/ModernBERT-large": 28,
        # # ...
        # "Qwen/Qwen3-0.6B-Base": 28,
        # "Qwen/Qwen3-0.6B": 28,
        # "Qwen/Qwen3-1.7B-Base": 28,
        # "Qwen/Qwen3-1.7B": 28,
        # "Qwen/Qwen3-4B-Base": 36,
        # "Qwen/Qwen3-4B": 36,
        # "Qwen/Qwen3-8B-Base": 36,
        # "Qwen/Qwen3-8B": 36,
        # "Qwen/Qwen3-Embedding-0.6B": 28,
        # "Qwen/Qwen3-Embedding-4B": 36,
        # "Qwen/Qwen3-Embedding-8B": 36,


        # to put BACK
        # "meta-llama/Llama-3.2-1B": 16,
        # "meta-llama/Llama-3.2-3B": 24,
        # "mistralai/Mistral-7B-v0.3": 32,
        # "allenai/OLMo-2-1124-7B": 32,
        # "google/gemma-2b": 18,
        # "google/gemma-7b": 28,
        # "google/gemma-2-2b": 26,
        # "google/gemma-3-1b-pt": 26,
    }

    models = [
        # ('dangvantuan/sentence-camembert-large', 24),
        # ('Lajavaness/sentence-camembert-large', 17),

        # ('almanach/camembert-large', 1),
        # ...
        # ('almanach/camembert-large', 24),

        # ('Qwen/Qwen3-Embedding-0.6B', None),
        # ('intfloat/multilingual-e5-large-instruct', None),
        # ('Qwen/Qwen3-Embedding-8B', None),
        # ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', None),
        # ('EuroBERT/EuroBERT-2.1B', None),
        # ('EuroBERT/EuroBERT-210m', None),
        # ('answerdotai/ModernBERT-base', None),
        # ('answerdotai/ModernBERT-large', None),
        # ('BAAI/bge-m3', None),
        # ('Qwen/Qwen3-0.6B', 28),
        # ('Qwen/Qwen3-4B', 36),
        # ('Qwen/Qwen3-8B', 36),
        # ('Qwen/Qwen3-Embedding-4B', 36),
    ]

    for model_name, num_layers in model2layer.items():
        for n_layers in range(num_layers):
            models.append((model_name, n_layers))

    # # adding Qwen/Qwen3-Embedding-8B from 29 to 36
    # for i in range(29, 37):
    #     models.append((f'Qwen/Qwen3-Embedding-8B', i))
    # adding Qwen/Qwen3-Embedding-8B-Base from 29 to 36
    # for i in range(29, 37):
    #     models.append((f'Qwen/Qwen3-8B-Base', i))

    # for i in range(10, 29):
    #     # if i not in [17, 18, 19, 20]:
    #         # models.append((f'Lajavaness/sentence-camembert-large', i))
    #     models.append((f'Qwen/Qwen3-Embedding-0.6B', i))

    for model_name, num_layers in models:
        print(f"Evaluating {model_name} with num_layers={num_layers}...")
        main(model_name, num_layers=num_layers)