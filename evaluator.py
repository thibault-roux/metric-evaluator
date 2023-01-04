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

    

def ember(ref, hyp, memory):
    tok2emb, threshold = memory
    
    erreurs = []
    ref = ref.split(" ")
    hyp = hyp.split(" ")
    list = awer.wer(ref, hyp)[0]
    # list = ['s', 'e', 'e', 's', 'e', 'i']
    for i in range(len(list)):
        element = list[i]
        if element == 's':
            # compute cosine similarity
            try:
                sim = 1 - spatial.distance.cosine(tok2emb[ref[i]], tok2emb[h[i]])
            except KeyError:
                sim = 0
            # threshold check
            if sim > threshold:
                erreurs.append(0.1)
            else:
                erreurs.append(1)
        elif element != "e":
            erreurs.append(1)
        else:
            erreurs.append(0)
    return sum(erreurs)/len(ref)


def custom_metric(ref, hyp, memory):
    return semdist(ref, hyp, memory)

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
        if c <= certitude or c >= 1-certitude: # if humans are certain about choice
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

    print("ratio correct:", correct/(correct+incorrect)*100)
    print("correct:", correct)
    print("incorrect:", incorrect)
    print("ratio ignored:", ignored/(ignored+accepted)*100)
    print("ignored:", ignored)
    print("accepted:", accepted)

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

    # semdist
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    memory=model
    """

    # ember
    from scipy import spatial
    import aligned_wer as awer
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


    # evaluation of metric
    print("Evaluation...")
    evaluator(custom_metric, dataset, memory=(tok2emb, 0), certitude=0)
    evaluator(custom_metric, dataset, memory=(tok2emb, 0), certitude=0.3)
    evaluator(custom_metric, dataset, memory=(tok2emb, 0), certitude=0.5)