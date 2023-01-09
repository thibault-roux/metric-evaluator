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


"""
def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score)*100 # lower is better
"""

"""
def semdist_bert(ref, hyp, memory):
    tokenizer, bert = memory
    ref_projection = bert(torch.tensor([tokenizer.encode(ref)]))[0][0].detach().numpy() #.reshape(1, -1)
    hyp_projection = bert(torch.tensor([tokenizer.encode(hyp)]))[0][0].detach().numpy() #.reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    
    return (1-score)*100 # lower is better
"""

"""
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
"""

"""
def wer_(ref, hyp, memory):
    return wer(ref, hyp)
"""

"""
def cer_(ref, hyp, memory):
    return cer(ref, hyp)
"""

"""
def bertscore(ref, hyp, memory):
    scorer = memory
    P, R, F1 = scorer.score([hyp], [ref])
    return 100-F1*100
"""

def character_bert(ref, hyp, memory):
    tokenizer, indexer, model = memory

    reference = ref # reference embeddings computation
    reference = ['[CLS]', *reference, '[SEP]']
    batch = [reference]
    batch_ids = indexer.as_padded_tensor(batch)
    embeddings_for_batch, _ = model(batch_ids)
    embeddings_for_ref = embeddings_for_batch[0]

    hypothesis = hyp # hypothesis embeddings computation
    hypothesis = ['[CLS]', *hypothesis, '[SEP]']
    batch = [hypothesis]
    batch_ids = indexer.as_padded_tensor(batch)
    embeddings_for_batch, _ = model(batch_ids)
    embeddings_for_hyp = embeddings_for_batch[0]

    score = cosine_similarity(embeddings_for_ref[0].detach().reshape(1, -1), embeddings_for_hyp[-1].detach().reshape(1, -1))[0][0]
    return score

def custom_metric(ref, hyp, memory):
    # return semdist(ref, hyp, memory)
    # return ember(ref, hyp, memory)
    # return semdist_bert(ref, hyp, memory) # FlauBERT & CamemBERT
    # return wer_(ref, hyp, memory)
    # return cer_(ref, hyp, memory)
    # return bertscore(ref, hyp, memory)
    return character_bert(ref, hyp, memory)

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
        try:
            c = nbrA/(nbrA+nbrB) # useful for human certitude of choice
        except ZeroDivisionError:
            ignored += 1
            continue
        if (c <= certitude or c >= 1-certitude) and nbrA+nbrB >= 5: # if humans are certain about choice
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

    print(" ratio correct:", correct/(correct+incorrect)*100)
    # print("correct:", correct)
    # print("incorrect:", incorrect)
    print("ratio ignored:", ignored/(ignored+accepted)*100)
    # print("ignored:", ignored)
    # print("accepted:", accepted)

    # 0-7 ; 1-6 ; 2-5 ; 3-4 ;
    # 0%  ; 14% ; 29% ; 43%
    # 
    # 0-6 ; 1-5 ; 2-4
    # 0%  ; 20% ; 50%
    #
    # 0-5 ; 1-4 ; 2-3
    # 0%  ; 25% ; 33%


if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    # useful for the metric but we do not need to recompute every time
    print("Importing...")

    """
    # semdist
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("Loading model...")
    # model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    # model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    # model = SentenceTransformer('distiluse-base-multilingual-cased')
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    model = SentenceTransformer('all-distilroberta-v1')
    model = SentenceTransformer('all-MiniLM-L12-v2')
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    memory=model
    """

    # semdist flaubert
    """
    import torch
    from transformers import FlaubertModel, FlaubertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    # modelname = 'flaubert/flaubert_base_cased'
    modelname = 'flaubert/flaubert_large_cased'
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(flaubert_tokenizer, flaubert)
    """

    # semdist CamemBERT
    """
    import torch
    from transformers import CamembertModel, CamembertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    # modelname = 'camembert-base'
    # modelname = 'camembert/camembert-large'
    modelname = 'camembert/camembert-base-wikipedia-4gb'
    camembert, log = CamembertModel.from_pretrained(modelname, output_loading_info=True)
    camembert_tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lowercase=True)
    memory=(camembert_tokenizer, camembert)
    """


    # ember
    """
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
    """

    # wer
    """
    from jiwer import wer
    memory = 0
    """

    # cer
    """
    from jiwer import cer
    memory = 0
    """

    # bertscore
    """
    from bert_score import BERTScorer
    # memory = BERTScorer(lang="fr")
    # memory = BERTScorer(model_type="amazon/bort")
    # memory = BERTScorer(model_type="distilbert-base-multilingual-cased")
    memory = BERTScorer(model_type="microsoft/deberta-xlarge-mnli")
    """

    # character-bert
    from transformers import BertTokenizer
    from modeling.character_bert import CharacterBertModel
    from utils.character_cnn import CharacterIndexer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    indexer = CharacterIndexer()
    model = CharacterBertModel.from_pretrained('/users/troux/these/expe/metrics/character-bert/pretrained-models/general_character_bert')
    memory = (tokenizer, indexer, model)


    # evaluation of metric
    print("Evaluation...")
    
    evaluator(custom_metric, dataset, memory=memory, certitude=0)
    evaluator(custom_metric, dataset, memory=memory, certitude=0.3)
    
    # mettre ce foutu code en propre en rangeant ce qui n'est pas métrique dans utils