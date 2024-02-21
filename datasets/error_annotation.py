import random
import jiwer
import os


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
    # check first if file exists
    if not os.path.exists("error_annotation.txt"):
        return []


    with open("error_annotation.txt", "r", encoding="utf8") as file:
        error_annotation = []
        for line in file:
            ref, hyp, word = line[:-1].split("\t")
            error_annotation.append((ref, hyp, word))
    return error_annotation



def filter(ref, hyp):
    # remove token "euh" from ref
    ref = ref.replace("euh", "")
    hyp = hyp.replace("euh", "")
    ref = ref.replace("-", " ")
    hyp = hyp.replace("-", " ")
    ref = ref.replace("  ", " ")
    hyp = hyp.replace("  ", " ")
    ref = ref.replace("  ", " ")
    hyp = hyp.replace("  ", " ")

    lenref = len(ref.split())
    lenhyp = len(hyp.split())
    if lenref < 3 or lenhyp < 3 or lenref > 20 or lenhyp > 20:
        return False
    elif ref == hyp:
        return False
    elif jiwer.wer(ref, hyp)*lenref < 2:
        return False
    elif jiwer.cer(ref, hyp)*len(ref) < 2:
        return False
    else:
        return True


def print_example(ref, hyp):
    print()
    out = jiwer.process_words(ref, hyp)
    printed = str(jiwer.visualize_alignment(out, show_measures=False)).split("\n")
    print(printed[1])
    print(printed[2])
    print(printed[3])
    print()
    print("Write << NONE >> when you don't know what to answer.")
    print("Enter the worst error pair: ")
    print()

def tutorial():
    print("Salut! Merci de venir m'aider. Je te fais un petit tuto pour t'expliquer comment ça fonctionne.")
    input("Appuie sur Entrée pour continuer...")

    ref = "un oiseau s'est envolé"
    hyp = "un eau s'est envolée"
    print_example(ref, hyp)
    print("Dans l'exemple ci-dessus, on te demande de taper la paire avec la plus grave erreur.")
    print("Ici, la pire erreur est la substitution de << oiseau >> par << eau >>.")
    print()
    input("Appuie sur Entrée pour continuer...")
    print("Il faut donc écrire:")
    print("Enter the worst error pair: oiseau/pair")
    print()
    input("Appuie sur Entrée pour continuer...")

    print("\n------------\n\n")
    print("Deuxième exemple !")
    print()

    ref = "hey salut tu es prêt pour le sport"
    hyp = "salut tu tu es pret pour le"
    print_example(ref, hyp)
    print("Dans cet exemple, la pire erreur est la suppression de << sport >>. Il faut écrire une '*'")
    print()
    input("Appuie sur Entrée pour continuer...")
    print("Il faut donc écrire:")
    print("Enter the worst error pair: sport/*")
    print()
    input("Appuie sur Entrée pour continuer...")

    print()
    print("N'oublie pas que l'on peut faire pareil pour les insertions !")

    
    print("\n------------\n\n")
    print("Tu dois être prêt maintenant. Bonne chance!")
    input()
    print("\n\n\n")


def annotation(skipped_tuto):
    if not skipped_tuto:
        tutorial()
    hats = load_hats()
    
    # shuffle hats
    random.shuffle(hats)

    error_annotation = load_annotated()
    

    with open("error_annotation.txt", "a", encoding="utf8") as file:
        for ref, hyp in hats:
            if (ref, hyp) not in error_annotation:
                if filter(ref, hyp):
                    out = jiwer.process_words(ref, hyp)
                    printed = str(jiwer.visualize_alignment(out, show_measures=False)).split("\n")
                    print()
                    print(printed[1])
                    print(printed[2])
                    print(printed[3])
                    print()
                    print("Write << NONE >> when you don't know what to answer.")
                    word = input("Enter the worst error pair: ")
                    if word == "NONE":
                        continue
                    else:
                        file.write(ref + "\t" + hyp + "\t" + word + "\n")
                        error_annotation.append((ref, hyp, word))




if __name__ == "__main__":
    skipped_tuto = True
    annotation(skipped_tuto)

    # il faut retirer les tirêts et les "euh"