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


def wer_(ref, hyp, memory):
    return wer(ref, hyp)


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
    return correct/(correct+incorrect+egal)*100


def write(namefile, x, y):
    with open("results/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + "," + str(x) + "," + str(y) + "\n")



if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1
    cert_Y = 0.7

    # useful for the metric but we do not need to recompute every time
    print("Importing...")


    
    from jiwer import wer
    evaluator(wer_, dataset, certitude=cert_X)
    evaluator(wer_, dataset, certitude=cert_Y)
    

    
    # semdist
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    
    # SD_sentence_camembert_large
    model = SentenceTransformer('dangvantuan/sentence-camembert-large')
    memory=model
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    """
    write("SD_sentence_camembert_large", x_score, y_score)
    """

    """
    # SD_sentence_camembert_base
    model = SentenceTransformer('dangvantuan/sentence-camembert-base')
    memory=model
    x_score = evaluator(semdist, dataset, memory=memory, certitude=cert_X)
    y_score = evaluator(semdist, dataset, memory=memory, certitude=cert_Y)
    #write("SD_sentence_camembert_base", x_score, y_score)
    """
    
    