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


def evaluator(metric1, metric2, dataset, memory1, memory2, verbose=True):
    # si > les deux, ou <, ou ==
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0

    if verbose:
        bar = progressbar.ProgressBar(max_value=len(dataset))
    for i in range(len(dataset)):
        if verbose:
            bar.update(i)
        
            score1A = metric1(dataset[i]["reference"], dataset[i]["hypA"], memory=memory1)
            score1B = metric1(dataset[i]["reference"], dataset[i]["hypB"], memory=memory1)
            score2A = metric2(dataset[i]["reference"], dataset[i]["hypA"], memory=memory2)
            score2B = metric2(dataset[i]["reference"], dataset[i]["hypB"], memory=memory2)
            if (score1A < score1B and score2A < score2B) or (score1A > score1B and score2A > score2B) or (score1A == score1B and score2A == score2B):
                correct += 1
            else:
                incorrect += 1
            continue
        else:
            ignored += 1

    print()
    print("ratio correct:", correct/(correct+incorrect)*100)
    return correct/(correct+incorrect)*100


def write(namefile, x):
    with open("results/agreement/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + "," + str(x) + "\n")

if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    # useful for the metric but we do not need to recompute every time
    print("Importing...")



    # dictionaries of models and memory
    Dicomodels = dict()
    Dicomemory = dict()


    # wer
    from jiwer import wer
    memory = 0
    Dicomodels["wer"] = wer_
    Dicomemory["wer"] = memory
    
    # cer
    from jiwer import cer
    memory2 = 0
    Dicomodels["wer"] = wer_
    Dicomemory["wer"] = memory



    # semdist ------------------
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD original # sentence
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    memory=model
    Dicomodels["SD_sentence_original"] = semdist
    Dicomemory["SD_sentence_original"] = memory
    
    # SD_sentence_camembert_large
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    memory=model
    Dicomodels["SD_sentence_camembert_large"] = semdist
    Dicomemory["SD_sentence_camembert_large"] = memory

    # SD_sentence_camembert_base
    model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    memory=model
    Dicomodels["SD_sentence_camembert_base"] = semdist
    Dicomemory["SD_sentence_camembert_base"] = memory


    # semdist flaubert ------
    import torch
    from transformers import FlaubertModel, FlaubertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # SD_flaubert_base
    modelname = 'flaubert/flaubert_base_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)
    Dicomodels["SD_flaubert_base"] = semdist_flaubert
    Dicomemory["SD_flaubert_base"] = memory
    
    # SD_flaubert_large
    modelname = 'flaubert/flaubert_large_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)
    Dicomodels["SD_flaubert_large"] = semdist_flaubert
    Dicomemory["SD_flaubert_large"] = memory


    # semdist camembert ---------
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
    Dicomodels["SD_camembert_base"] = semdist_camembert
    Dicomemory["SD_camembert_base"] = memory

    # SD_camembert_large
    modelname = 'camembert/camembert-large'
    config = CamembertConfig.from_pretrained(modelname, output_hidden_states=True, return_dict=False) # added
    camembert = CamembertModel.from_pretrained(modelname, config=config) # added
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert, -1)
    Dicomodels["SD_camembert_large"] = semdist_camembert
    Dicomemory["SD_camembert_large"] = memory

    
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
    Dicomodels["ember"] = ember
    Dicomemory["ember"] = memory
    
    
    # bertscore
    from bert_score import BERTScorer

    # BS
    memory = BERTScorer(lang="fr")
    Dicomodels["BS"] = bertscore
    Dicomemory["BS"] = memory


    # BS_camembert-base
    memory = BERTScorer(model_type="camembert-base", num_layers=12)
    Dicomodels["BS_camembert-base"] = bertscore
    Dicomemory["BS_camembert-base"] = memory

    # BS_camembert-large
    memory = BERTScorer(model_type="camembert/camembert-large", num_layers=24)
    Dicomodels["BS_camembert-large"] = bertscore
    Dicomemory["BS_camembert-large"] = memory
    

    for m1, _ in Dicomodels.items():
        for m2, _ in Dicomodels.items():
            if m1 != m2:
                print(m1, m2)
                x_score = evaluator(Dicomodels[m1], Dicomodels[m2], dataset, memory1=Dicomemory[m1], memory2=Dicomemory[m2])
                write(m1 + "_" + m2, x_score)
