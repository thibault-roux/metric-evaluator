import progressbar


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

def custom_metric(ref, hyp, memory=0):
    # compute a score given a textual reference and hypothesis
    return score

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

    # useful for the metric but we do not need to recompute every time
    print("Importing...")

    """
    # semdist
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD original # sentence
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    memory=model
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    write("SD_sentence_original", x_score, y_score)

    # SD_sentence_camembert_large
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    memory=model
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    write("SD_sentence_camembert_large", x_score, y_score)

    # SD_sentence_camembert_base
    model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    memory=model
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    write("SD_sentence_camembert_base", x_score, y_score)
    
    
    

    # semdist flaubert
    import torch
    from transformers import FlaubertModel, FlaubertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD_flaubert_base
    modelname = 'flaubert/flaubert_base_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)
    x_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_Y)
    write("SD_flaubert_base", x_score, y_score)
    
    # SD_flaubert_large
    modelname = 'flaubert/flaubert_large_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)
    x_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_Y)
    write("SD_flaubert_large", x_score, y_score)


    # semdist camembert
    import torch
    from transformers import CamembertModel, CamembertTokenizer
    from transformers import CamembertConfig # added
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD_camembert_base
    modelname = 'camembert-base'
    config = CamembertConfig.from_pretrained(modelname, output_hidden_states=True, return_dict=False) # added
    camembert = CamembertModel.from_pretrained(modelname, config=config) # added
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert, -1)
    x_score = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Y)
    write("SD_camembert_base", x_score, y_score)

    
    # SD_camembert_large
    modelname = 'camembert/camembert-large'
    config = CamembertConfig.from_pretrained(modelname, output_hidden_states=True, return_dict=False) # added
    camembert = CamembertModel.from_pretrained(modelname, config=config) # added
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert, -1)
    x_score = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Y)
    write("SD_camembert_large", x_score, y_score)


    # ember
    from scipy import spatial
    import utils.aligned_wer as awer
    import numpy as np
    vocabulary = set()
    for d in dataset:
        for element in ["reference", "hypA", "hypB"]:
            words = d[element].split(" ")
            vocabulary.update(words)

    namefile = "utils/embeddings/cc.fr.300.vec"
    print("Embeddings loading...")
    tok2emb = {}
    file = open(namefile, "r", encoding="utf8")
    next(file)
    for ligne in file:
        ligne = ligne[:-1].split(" ")
        if ligne[0] in vocabulary:
            emb = np.array(ligne[1:]).astype(float)
            if emb.shape != (300,):
                print("Erreur à " + ligne[0])
            else:
                tok2emb[ligne[0]] = emb
    print("Embeddings loaded.")
    memory=(tok2emb, 0.3, 0.4)
    x_score = evaluator(ember, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(ember, dataset, memory=memory, certitude=cert_Y)
    write("ember", x_score, y_score)
    

    # wer
    from jiwer import wer
    memory = 0
    x_score = evaluator(wer_, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(wer_, dataset, memory=memory, certitude=cert_Y)
    write("wer", x_score, y_score)

     # cer
    from jiwer import cer
    memory = 0
    x_score = evaluator(cer_, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(cer_, dataset, memory=memory, certitude=cert_Y)
    write("cer", x_score, y_score)

    

    # bertscore
    from bert_score import BERTScorer

    # BS
    memory = BERTScorer(lang="fr")
    x_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_Y)
    write("BS", x_score, y_score)

    # BS_camembert-base
    memory = BERTScorer(model_type="camembert-base", num_layers=12)
    x_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_Y)
    write("BS_camembert-base", x_score, y_score)

    # BS_camembert-large
    memory = BERTScorer(model_type="camembert/camembert-large", num_layers=24)
    x_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(bertscore, dataset, memory=memory, certitude=cert_Y)
    write("BS_camembert-large", x_score, y_score)

    """

    # SD_bloom
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modelname = 'bigscience/bloom'
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", torch_dtype="auto")
    memory=(tokenizer, model)
    x_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_Y)
    write("SD_bloom", x_score, y_score)
    x_score = evaluator(custom_metric, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(custom_metric, dataset, memory=memory, certitude=cert_Y)
