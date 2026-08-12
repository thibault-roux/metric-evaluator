import progressbar
import csv
import os
from pathlib import Path

# Ordre de qualité des classes (du meilleur au pire)
CLASS_QUALITY = {
    "identique": 4,
    "utile": 3,
    "mauvaise": 2,
    "incompréhensible": 1
}

def read_annotations(annotation_file):
    """
    Lit les annotations d'un fichier CSV.
    Retourne un dictionnaire: {(ref, hyp): class}
    """
    annotations = {}
    with open(annotation_file, "r", encoding="utf8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ref = row["ref"].strip()
            hyp = row["hyp"].strip()
            class_label = row["class"].strip()
            # Utiliser (ref, hyp) comme clé
            annotations[(ref, hyp)] = class_label
    return annotations

def get_best_hypothesis(hyp_a, class_a, hyp_b, class_b):
    """
    Retourne l'hypothèse la meilleure basée sur la qualité de classe.
    Retourne: ("A" ou "B", quality_a, quality_b, True si égalité)
    """
    quality_a = CLASS_QUALITY.get(class_a, 0)
    quality_b = CLASS_QUALITY.get(class_b, 0)
    
    if quality_a > quality_b:
        return "A", quality_a, quality_b, False
    elif quality_b > quality_a:
        return "B", quality_a, quality_b, False
    else:
        return "equal", quality_a, quality_b, True

def read_dataset(dataname):
    """
    Lit le fichier dataset au format TSV.
    Retourne une liste de dictionnaires.
    """
    dataset = []
    with open(dataname, "r", encoding="utf8") as file:
        next(file)  # Skip header
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

def evaluator(annotations, dataset, certitude=0.3, verbose=True):
    """
    Évalue les annotations en les comparant au dataset.
    Retourne les statistiques d'accord et d'égalité.
    """
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0
    equal = 0
    not_found = 0
    
    if verbose:
        bar = progressbar.ProgressBar(max_value=len(dataset))
    
    for i in range(len(dataset)):
        if verbose:
            bar.update(i)
        
        ref = dataset[i]["reference"]
        hypA = dataset[i]["hypA"]
        hypB = dataset[i]["hypB"]
        nbrA = dataset[i]["nbrA"]
        nbrB = dataset[i]["nbrB"]
        
        # Vérifier la certitude humaine
        if nbrA + nbrB < 5:
            ignored += 1
            continue
        
        maximum = max(nbrA, nbrB)
        c = maximum / (nbrA + nbrB)
        
        if c < certitude:  # Si les humains ne sont pas certains
            ignored += 1
            continue
        
        # Déterminer le choix humain (référence)
        human_choice = "A" if nbrA > nbrB else "B"
        
        # Chercher les annotations
        annotation_a = annotations.get((ref, hypA))
        annotation_b = annotations.get((ref, hypB))
        
        if annotation_a is None or annotation_b is None:
            not_found += 1
            ignored += 1
            continue
        
        accepted += 1
        
        # Obtenir la meilleure hypothèse selon les annotations
        best, qual_a, qual_b, is_equal = get_best_hypothesis(
            hypA, annotation_a, hypB, annotation_b
        )
        
        if is_equal:
            equal += 1
        elif best == human_choice:
            correct += 1
        else:
            incorrect += 1
    
    if verbose:
        print()
    
    total_decisions = correct + incorrect
    
    # Calcul des pourcentages
    if total_decisions > 0:
        ratio_correct = correct / total_decisions * 100
    else:
        ratio_correct = 0
    
    if accepted > 0:
        ratio_correct_absolute = correct / accepted * 100
        ratio_equal = equal / accepted * 100
    else:
        ratio_correct_absolute = 0
        ratio_equal = 0
    
    ratio_ignored = ignored / (ignored + accepted) * 100 if (ignored + accepted) > 0 else 0
    
    # Affichage
    print(f"Éléments acceptés: {accepted}")
    print(f"Éléments ignorés: {ignored}")
    print(f"Non trouvés dans annotations: {not_found}")
    print(f"Égalités (même classe): {equal}")
    print(f"Accord correct: {correct}")
    print(f"Accord incorrect: {incorrect}")
    print(f"Ratio accord (sur décisions uniquement): {ratio_correct:.2f}%")
    print(f"Ratio accord absolu (sur acceptés): {ratio_correct_absolute:.2f}%")
    print(f"Ratio égalité: {ratio_equal:.2f}%")
    print(f"Ratio ignoré: {ratio_ignored:.2f}%")
    
    return ratio_correct, ratio_correct_absolute, ratio_equal, ratio_ignored

def extract_model_name(annotation_file):
    """
    Extrait le nom du modèle du nom du fichier.
    Exemple: "results/classified_output_gemma3-27b.csv" -> "gemma3-27b"
    """
    filename = Path(annotation_file).stem  # Récupère le nom sans extension
    # Extrait la partie après "classified_output_"
    if "classified_output_" in filename:
        return filename.split("classified_output_")[-1]
    return filename

if __name__ == '__main__':
    # Configuration
    # annotation_file = "./results/classified_output_gemma3-27b.csv"
    annotation_file = input("Enter the path to the annotation CSV file (e.g., results/classified_output_gemma3-27b.csv): ")
    dataset_file = "datasets/hats.txt"
    output_file = "./out/results_class.txt"
    
    cert_levels = [
        (1.0, "X"),
        (0.7, "Y"),
        (0.0, "Z")
    ]
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Vérifier que les fichiers existent
    if not os.path.exists(annotation_file):
        print(f"Erreur: Fichier d'annotations non trouvé: {annotation_file}")
        exit(1)
    
    if not os.path.exists(dataset_file):
        print(f"Erreur: Fichier dataset non trouvé: {dataset_file}")
        exit(1)
    
    # Lire les données
    print("Lecture des annotations...")
    annotations = read_annotations(annotation_file)
    print(f"  {len(annotations)} paires (ref, hyp) chargées")
    
    print("Lecture du dataset...")
    dataset = read_dataset(dataset_file)
    print(f"  {len(dataset)} éléments chargés")
    
    # Extraire le nom du modèle
    model_name = extract_model_name(annotation_file)
    print(f"\nModèle: {model_name}\n")
    
    # Évaluation pour différents niveaux de certitude
    results = []
    for certitude, label in cert_levels:
        print(f"=== Évaluation avec certitude = {certitude} ({label}) ===")
        ratio_correct, ratio_correct_absolute, ratio_equal, ratio_ignored = evaluator(
            annotations, dataset, certitude=certitude, verbose=True
        )
        results.append((certitude, ratio_correct, ratio_correct_absolute, ratio_equal, ratio_ignored))
        print()
    
    # Enregistrer les résultats
    print("Enregistrement des résultats...")
    with open(output_file, "a", encoding="utf8") as file:
        for certitude, ratio_correct, ratio_correct_absolute, ratio_equal, ratio_ignored in results:
            file.write(
                f"{model_name}\t{certitude}\t"
                f"{ratio_correct:.2f}\t{ratio_correct_absolute:.2f}\t"
                f"{ratio_equal:.2f}\t{ratio_ignored:.2f}\n"
            )
    
    print(f"Résultats sauvegardés dans: {output_file}")