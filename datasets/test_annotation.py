import align



def load_annotation(namefile):
    with open(namefile, "r", encoding="utf8") as file:
        error_annotation = []
        for line in file:
            ref, hyp, pair = line[:-1].split("\t")
            pair = pair.split("/")
            error_annotation.append((ref, hyp, pair))
    return error_annotation


def semdist(ref, hyp, memory):
    model = memory
    ref_projection = model.encode(ref).reshape(1, -1)
    hyp_projection = model.encode(hyp).reshape(1, -1)
    score = cosine_similarity(ref_projection, hyp_projection)[0][0]
    return (1-score) # lower is better

def cer(ref, hyp, memory):
    return jiwer.cer(ref, hyp)


if __name__ == "__main__":
    data = load_annotation("error_annotation.txt")


    
    choice = "cer"
    # choice = "SD_sent_camemlarge"

    if choice == "cer":
        import jiwer
        memory = 0
        metric = cer
    elif choice == "SD_sent_camemlarge":
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer('dangvantuan/sentence-camembert-large')
        memory = model
        metric = semdist
    else:
        raise Exception("Unknown choice: ", choice)
    

    accuracy = 0
    for i in range(len(data)):
        ref, hyp, pair = data[i]
        errors, _, alignment_ref, alignment_hup = align.awer(ref, hyp, return_alignments=True)

        semdist_scores = []
        index_corrected= []
        for j in range(len(errors)):
            if errors[j] != "e":
                hyp = correct(alignment_ref, alignment_hyp, j, errors[j])
                scores.append(semdist(ref, hyp, model))
                index_corrected.append(j)
        # get the index of the lowest score
        if len(scores) > 0:
            index = scores.index(min(scores)) # index in the scores list
            best = index_corrected[index] # index in the alignment
            if alignment_ref[best] == pair[0] and alignment_hyp[best] == pair[1]:
                accuracy += 1
        else:
            raise ValueError("No scores found")

    print("Accuracy: ", accuracy/len(data))
