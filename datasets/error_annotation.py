import random



def load_hats():
    with open("hats.txt", "r", encoding="utf8") as file:
        next(file)
        hats = []
        for line in file:
            ref, hypA, nbrA, hypB, nbrB = line[:-1].split("\t")
            hats.append((ref, hypA))
            hats.append((ref, hypB))
    return hats

def load_annotated():
    with open("error_annotation.txt", "r", encoding="utf8") as file:
        error_annotation = []
        for line in file:
            ref, hyp, word = line[:-1].split("\t")
            error_annotation.append((ref, hyp, word))
    return error_annotation


if __name__ == "__main__":
    hats = load_hats()
    
    # shuffle hats
    random.shuffle(hats)

    error_annotation = load_annotated()

    with open("error_annotation.txt", "a", encoding="utf8") as file:
        for ref, hyp in hats:
            if (ref, hyp) not in error_annotation:
                if filter(ref, hyp):
                    print("ref:", ref)
                    print("hyp:", hyp)
                    word = input("Enter the reference word with the worst error: ")
                    file.write(ref + "\t" + hyp + "\t" + word + "\n")
                    error_annotation.append((ref, hyp, word))