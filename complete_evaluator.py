import progressbar
import numpy


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


def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better


def semdist_flaubert(ref, hyp, memory):
    flaubert_tokenizer, flaubert = memory
    ref_projection = flaubert(torch.tensor([flaubert_tokenizer.encode(ref)]))[0][0].detach().numpy() #.reshape(1, -1)
    hyp_projection = flaubert(torch.tensor([flaubert_tokenizer.encode(hyp)]))[0][0].detach().numpy() #.reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    
    return (1-score)*100 # lower is better

def semdist_camembert(ref, hyp, memory):
    camembert_tokenizer, camembert, layer = memory

    encoded_ref = torch.tensor(camembert_tokenizer.encode(camembert_tokenizer.tokenize(ref))).unsqueeze(0)
    encoded_hyp = torch.tensor(camembert_tokenizer.encode(camembert_tokenizer.tokenize(hyp))).unsqueeze(0)
    _, _, all_layer_embeddings_ref = camembert(encoded_ref)
    _, _, all_layer_embeddings_hyp = camembert(encoded_hyp)
    ref_projection = all_layer_embeddings_ref[layer][0].detach().numpy()
    hyp_projection = all_layer_embeddings_hyp[layer][0].detach().numpy()
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    
    return (1-score)*100 # lower is better


def ember(ref, hyp, memory):
    tok2emb, threshold, weight = memory
    
    erreurs = []
    ref = ref.split(" ")
    hyp = hyp.split(" ")
    list = awer.wer(ref, hyp)[0]
    # list = ['s', 'e', 'e', 's', 'e', 'i']
    ri = 0 # indice for ref
    hi = 0
    for i in range(len(list)):
        element = list[i]
        if element == 's':
            # compute cosine similarity
            try:
                sim = 1 - spatial.distance.cosine(tok2emb[ref[ri]], tok2emb[hyp[hi]])
            except KeyError:
                sim = 0 # one of words do not exist in the vocabulary
            ri += 1
            hi += 1
            # threshold check
            if sim > threshold:
                erreurs.append(weight)
            else:
                erreurs.append(1)
        elif element != "e":
            erreurs.append(1)
            if element == "d":
                ri += 1
            elif element == "i":
                hi += 1
            else:
                print("ERREUR, element == " + element)
                exit(-1)
        else:
            erreurs.append(0)
            ri += 1
            hi += 1
    return sum(erreurs)/len(ref)

def wer_(ref, hyp, memory):
    return wer(ref, hyp)

def cer_(ref, hyp, memory):
    return cer(ref, hyp)

def bertscore(ref, hyp, memory):
    scorer = memory
    P, R, F1 = scorer.score([hyp], [ref])
    return 100-F1*100

def evaluator(metric, dataset, memory=0, certitude=0.7, verbose=True):
    print("certitude: ", certitude*100)
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0
    egal = 0

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
            elif scoreA == scoreB:
                egal += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1
    print()
    print("correct:", correct)
    print("incorrect:", incorrect)
    print("egal:", egal)
    print("ratio correct:", correct/(correct+incorrect+egal)*100)
    print("ratio egal: ", egal/(correct+incorrect+egal)*100)
    print()
    print("ratio ignored:", ignored/(ignored+accepted)*100)
    print("ignored:", ignored)
    print("accepted:", accepted)
    return correct, egal, incorrect
    # return correct/(correct+incorrect+egal)*100


def tostring(tuple):
    return str(tuple[0]) + "," + str(tuple[1]) + "," + str(tuple[2])
def write(namefile, x, y, z):
    with open("results/complete/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + ";" + tostring(x) + ";" + tostring(y) + ";" + tostring(z) + "\n")



if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1
    cert_Y = 0.7
    cert_Z = 0

    # useful for the metric but we do not need to recompute every time
    print("Importing...")


    
    """from jiwer import wer
    x = evaluator(wer_, dataset, certitude=cert_X)
    y = evaluator(wer_, dataset, certitude=cert_Y)
    z = evaluator(wer_, dataset, certitude=cert_Z) # full dataset
    write("wer", x, y, z)"""


    """from jiwer import cer
    x = evaluator(cer_, dataset, certitude=cert_X)
    y = evaluator(cer_, dataset, certitude=cert_Y)
    z = evaluator(cer_, dataset, certitude=cert_Z) # full dataset
    write("cer", x, y, z)"""



    """# semdist
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    
    # SD_sentence_original
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    memory=model
    x = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(semdist, dataset, memory=memory, certitude=cert_Z) # full dataset
    write("SD_sentence_original", x, y, z)"""


    """
    # bertscore
    
    from bert_score import BERTScorer
    # memory = BERTScorer(lang="fr")
    # memory = BERTScorer(model_type="amazon/bort")
    # memory = BERTScorer(model_type="distilbert-base-multilingual-cased")
    # memory = BERTScorer(model_type="microsoft/deberta-xlarge-mnli")
    # memory = BERTScorer(model_type="camembert-base")
    memory = BERTScorer(model_type="camembert/camembert-large")
    x = evaluator(bertscore, dataset, memory=memory, certitude=cert_X)
    y = evaluator(bertscore, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(bertscore, dataset, memory=memory, certitude=cert_Z)
    write("BS_camembert-large", x, y, z)

    from bert_score import BERTScorer
    # memory = BERTScorer(lang="fr")
    # memory = BERTScorer(model_type="amazon/bort")
    # memory = BERTScorer(model_type="distilbert-base-multilingual-cased")
    # memory = BERTScorer(model_type="microsoft/deberta-xlarge-mnli")
    memory = BERTScorer(model_type="camembert-base")
    # memory = BERTScorer(model_type="camembert/camembert-large")
    x = evaluator(bertscore, dataset, memory=memory, certitude=cert_X)
    y = evaluator(bertscore, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(bertscore, dataset, memory=memory, certitude=cert_Z)
    write("BS_camembert-base", x, y, z)
    """





    """# ember
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


    x = evaluator(ember, dataset, memory=memory, certitude=cert_X)
    y = evaluator(ember, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(ember, dataset, memory=memory, certitude=cert_Z)
    write("ember", x, y, z)"""
    
    
    
    """
    # semdist flaubert ------
    import torch
    from transformers import FlaubertModel, FlaubertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD_flaubert_base
    modelname = 'flaubert/flaubert_base_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)

    x = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_X)
    y = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(semdist_flaubert, dataset, memory=memory, certitude=cert_Z)
    write("SD_flaubert_base", x, y, z)

    """




    # semdist camembert ---------
    import torch
    from transformers import CamembertModel, CamembertTokenizer
    from transformers import CamembertConfig # added
    from sklearn.metrics.pairwise import cosine_similarity
    
    """
    # SD_camembert_base
    modelname = 'camembert-base'
    config = CamembertConfig.from_pretrained(modelname, output_hidden_states=True, return_dict=False) # added
    camembert = CamembertModel.from_pretrained(modelname, config=config) # added
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert, -1)

    x = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_X)
    y = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Z)
    write("SD_camembert_base", x, y, z)
    """


    # SD_camembert_large
    modelname = 'camembert/camembert-large'
    config = CamembertConfig.from_pretrained(modelname, output_hidden_states=True, return_dict=False) # added
    camembert = CamembertModel.from_pretrained(modelname, config=config) # added
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert, -1)

    x = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_X)
    y = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Y)
    z = evaluator(semdist_camembert, dataset, memory=memory, certitude=cert_Z)
    write("SD_camembert_large", x, y, z)



    
    print("Fin")
    exit(-1)


    