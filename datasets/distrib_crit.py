import jiwer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np
import pickle



#----------------------- UTILS -----------------------#

def similarite(syn1, syn2):
    try:
        return 1 - spatial.distance.cosine(syn1, syn2)
    except KeyError:
        return 0



#----------------------- METRICS -----------------------#

def wer(ref, hyp):
    return jiwer.wer(ref, hyp)

def semdist(ref, hyp, memory):
    model = memory
    if ref != hyp:
        ref_proj = model.encode(ref).reshape(1, -1)
        hyp_proj = model.encode(hyp).reshape(1, -1)
        sd = cosine_similarity(ref_proj, hyp_proj)[0][0]
    else:
        sd = 1
    return (1 - sd)*100

    

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

    input("Continuer ?")


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
        bertscore_results.append([F1_A[i], F1_B[i]])

    # save results 
    with open("pickle/bertscore_results.pkl", "wb") as file:
        pickle.dump(bertscore_results, file)


    print("Computing WER...")
    werA = [wer(ref, hyp) for ref, hyp in zip(ref, hypsA)]
    werB = [wer(ref, hyp) for ref, hyp in zip(ref, hypsB)]
    wer_results = []
    for i in range(len(werA)):
        wer_results.append([werA[i], werB[i]])

    # save results 
    with open("pickle/wer_results.pkl", "wb") as file:
        pickle.dump(wer_results, file)


if __name__ == "__main__":
    compute_metrics()

    semdist_results = pickle.load(open("pickle/semdist_results.pkl", "rb"))
    bertscore_results = pickle.load(open("pickle/bertscore_results.pkl", "rb"))
    wer_results = pickle.load(open("pickle/wer_results.pkl", "rb"))

    print(len(semdist_results))
    print(len(bertscore_results))
    print(len(wer_results))