import jiwer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np
import pickle
from scipy import spatial



#----------------------- UTILS -----------------------#

def similarite(syn1, syn2):
    try:
        return 1 - spatial.distance.cosine(syn1, syn2)
    except KeyError:
        return 0

def load_embeddings(namefile, vocab):
    tok2emb = {}
    try:
        file = open(namefile, "r", encoding="utf8")
    except FileNotFoundError:
        print("ERROR: embeddings file " + str(namefile) + " is not found. Please download text embeddings from https://fasttext.cc/docs/en/crawl-vectors.html in utils/embeddings. If you already downloaded the file, make sure your filename matches the one in utils/eval.Ember().namefile")
        raise
    next(file)
    for ligne in file:
        ligne = ligne[:-1].split(" ")
        if ligne[0] in vocab:
            emb = np.array(ligne[1:]).astype(float)
            if emb.shape != (300,):
                print("Erreur à " + ligne[0])
            else:
                tok2emb[ligne[0]] = emb
    print("Embeddings loaded.")
    return tok2emb

def levenstein_alignment(ref, hyp):
    # create a matrix of size (len(ref)+1) x (len(hyp)+1)
    # the first row and the first column are filled with 0, 1, 2, 3, ...
    # the rest of the matrix is filled with -1
    matrix = np.zeros((len(ref)+1, len(hyp)+1))
    for i in range(1, len(ref)+1):
        matrix[i, 0] = i
    for j in range(1, len(hyp)+1):
        matrix[0, j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            matrix[i, j] = -1

    # fill the matrix with the correct values
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                matrix[i, j] = matrix[i-1, j-1]
            else:
                matrix[i, j] = min(matrix[i-1, j-1], matrix[i-1, j], matrix[i, j-1]) + 1

    # create two lists of words with a <epsilon> token for insertion and deletion
    ref_aligned = []
    hyp_aligned = []
    i = len(ref)
    j = len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ref_aligned.append(ref[i-1])
            hyp_aligned.append(hyp[j-1])
            i -= 1
            j -= 1
        elif i > 0 and matrix[i, j] == matrix[i-1, j] + 1:
            ref_aligned.append(ref[i-1])
            hyp_aligned.append("<epsilon>")
            i -= 1
        elif j > 0 and matrix[i, j] == matrix[i, j-1] + 1:
            ref_aligned.append("<epsilon>")
            hyp_aligned.append(hyp[j-1])
            j -= 1
        else:
            ref_aligned.append(ref[i-1])
            hyp_aligned.append(hyp[j-1])
            i -= 1
            j -= 1

    refa, hypa = ref_aligned[::-1], hyp_aligned[::-1]
    binary_list = [int(r == h) for r, h in zip(refa, hypa)]
    return refa, hypa, binary_list



#----------------------- METRICS -----------------------#

def wer(ref, hyp):
    return jiwer.wer(ref, hyp)

def cer(ref, hyp):
    return jiwer.cer(ref, hyp)

def semdist(ref, hyp, memory):
    model = memory
    if ref != hyp:
        ref_proj = model.encode(ref).reshape(1, -1)
        hyp_proj = model.encode(hyp).reshape(1, -1)
        sd = cosine_similarity(ref_proj, hyp_proj)[0][0]
    else:
        sd = 1
    return (1 - sd)*100


def ember(ref, hyp, tok2emb):
    threshold = 0.4
    voc = tok2emb.keys()
    errors = []
    d = 0
    c = 0
    error = 0
    r, h, binary_list = levenstein_alignment(ref.split(" "), hyp.split(" "))
    if len(r) != len(h): # if ref and hypothesis are different -> error
        d += 1
        raise Exception("ERROR: length different.")
    else:
        c += 1
    for j in range(len(r)):
        if r[j] != h[j]:
            if r[j] == "<epsilon>" or h[j] == "<epsilon>":
                error += 1
            else:
                if r[j] in voc and h[j] in voc:
                    sim = similarite(tok2emb[r[j]], tok2emb[h[j]])
                    if sim > threshold: # Threshold
                        error += 0.1
                    else:
                        error += 1
                else:
                    error += 1
        else:
            error += 0
    return error/len(r)








#----------------------- METRICS-TOTAL -----------------------#

def wer_global(refs, hypsA, hypsB):
    print("Computing WER...")
    werA = [wer(ref, hyp) for ref, hyp in zip(refs, hypsA)]
    werB = [wer(ref, hyp) for ref, hyp in zip(refs, hypsB)]
    wer_results = []
    for i in range(len(werA)):
        wer_results.append([werA[i], werB[i]])

    # save results 
    with open("pickle/wer_results.pkl", "wb") as file:
        pickle.dump(wer_results, file)

def cer_global(refs, hypsA, hypsB):
    print("Computing CER...")
    cerA = [cer(ref, hyp) for ref, hyp in zip(refs, hypsA)]
    cerB = [cer(ref, hyp) for ref, hyp in zip(refs, hypsB)]
    cer_results = []
    for i in range(len(cerA)):
        cer_results.append([cerA[i], cerB[i]])

    # save results 
    with open("pickle/cer_results.pkl", "wb") as file:
        pickle.dump(cer_results, file)


def ember_global(refs, hypsA, hypsB):
    # recover the vocabulary of refs, hypsA and hypsB
    vocab = set()
    for ref in refs:
        for word in ref.split(" "):
            vocab.add(word)
    for hyp in hypsA:
        for word in hyp.split(" "):
            vocab.add(word)
    for hyp in hypsB:
        for word in hyp.split(" "):
            vocab.add(word)
    tok2emb = load_embeddings("../../hypereval/utils/embeddings/cc.fr.300.vec", vocab) # FINIR CA

    print("Computing EmbER...")
    emberA = [ember(ref, hyp, tok2emb) for ref, hyp in zip(refs, hypsA)]
    emberB = [ember(ref, hyp, tok2emb) for ref, hyp in zip(refs, hypsB)]
    ember_results = []
    for i in range(len(emberA)):
        ember_results.append([emberA[i], emberB[i]])

    # save results 
    with open("pickle/ember_results.pkl", "wb") as file:
        pickle.dump(ember_results, file)


def semdist_global(refs, hypsA, hypsB):
    print("Computing SemDist...")
    semdist_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    semdist_results = []
    for i in range(len(refs)):
        ref = refs[i]
        hypA = hypsA[i]
        hypB = hypsB[i]
        semdist_results.append([semdist(ref, hypA, semdist_model), semdist(ref, hypB, semdist_model)])

    # save results 
    with open("pickle/semdist_results.pkl", "wb") as file:
        pickle.dump(semdist_results, file)

def bertscore_global(refs, hypsA, hypsB):
    print("Computing BertScore...")
    # nphyps = np.array(hyps)
    # P, R, F1_A = score(refs, nphyps[:,0], lang="fr", verbose=True)
    # P, R, F1_B = score(refs, nphyps[:,1], lang="fr", verbose=True)
    P, R, F1_A = score(refs, hypsA, lang="fr", verbose=True)
    P, R, F1_B = score(refs, hypsB, lang="fr", verbose=True)
    # write code to convert F1 to a list where each value is substracted by 1 and multiplied by 100
    F1_A = [((1 - f1) * 100) for f1 in F1_A]
    F1_B = [((1 - f1) * 100) for f1 in F1_B]
    
    bertscore_results = []
    for i in range(len(F1_A)):
        bertscore_results.append([F1_A[i].item(), F1_B[i].item()])
        
    # save results 
    with open("pickle/bertscore_results.pkl", "wb") as file:
        pickle.dump(bertscore_results, file)



    

def compute_metrics():
    with open("hats.txt", "r", encoding="utf8") as file:
        next(file)
        refs = []
        hypsA = []
        hypsB = []
        for line in file:
            line = line.strip().split("\t")
            refs.append(line[0])
            hypsA.append(line[1])
            hypsB.append(line[3])


    wer_global(refs, hypsA, hypsB)
    cer_global(refs, hypsA, hypsB)
    ember_global(refs, hypsA, hypsB)
    semdist_global(refs, hypsA, hypsB)
    bertscore_global(refs, hypsA, hypsB)
    


if __name__ == "__main__":
    # compute_metrics()

    semdist_results = pickle.load(open("pickle/semdist_results.pkl", "rb"))
    bertscore_results = pickle.load(open("pickle/bertscore_results.pkl", "rb"))
    wer_results = pickle.load(open("pickle/wer_results.pkl", "rb"))

    metrics = ["wer", "cer", "ember", "semdist", "bertscore"]
    opposite = dict()
    for metric1, metric2 in list(combinations(metrics, 2)):
        opposite[metric1 + "INV" + metric2] = 0
    
    for i in range(len(semdist_results)):
        sd1, sd2 = semdist_results[i]
        bs1, bs2 = bertscore_results[i]
        wr1, wr2 = wer_results[i]

        if (wr1 > wr2 and sd1 < sd2) or (wr1 < wr2 and sd1 > sd2):
            opposite["werINVsemdist"] += 1
