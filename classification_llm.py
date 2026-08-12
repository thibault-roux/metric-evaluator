import pandas as pd
import csv
import re
import ollama

INPUT_TXT = "datasets/hats.txt"
OUTPUT_CSV = "classified_output.csv"

PROMPT_TEMPLATE = """
Tu es un évaluateur expert en reconnaissance automatique de la parole (ASR).

Étant donné :
- Une RÉFÉRENCE (transcription correcte) : "{ref}"
- Une HYPOTHÈSE (transcription produite par un système ASR) : "{hyp}"

Classe l'hypothèse dans exactement UNE des catégories suivantes :

| Catégorie        | Critères                                                                                      |
|------------------|-----------------------------------------------------------------------------------------------|
| IDENTIQUE        | L'hypothèse est identique à la référence (ou différence de casse ou de tiret).               |
| UTILE            | Le sens est préservé malgré des erreurs mineures : normalisation, ponctuation, orthographe,  |
|                  | majuscules, abréviations, ou légères variations syntaxiques sans perte de compréhension.     |
| MAUVAISE         | Le sens est partiellement altéré : erreurs significatives sur des mots-clés, substitutions   |
|                  | ou omissions importantes, mais une partie du contenu reste compréhensible.                   |
| INCOMPRÉHENSIBLE | Le sens est totalement perdu : la phrase ne peut pas être comprise ou interprétée             |
|                  | correctement en lien avec la référence.                                                       |

Réponds UNIQUEMENT avec le format suivant, sans explication :
CATEGORIE: <catégorie>
"""

def classify(ref, hyp):
    prompt = PROMPT_TEMPLATE.format(ref=ref, hyp=hyp)

    response = ollama.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response["message"]["content"]
    print(text)
    match = re.search(r"(identique|utile|mauvaise|incompréhensible)", text, re.IGNORECASE)
    return match.group(1).lower() if match else "incompréhensible"

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_TXT, sep="\t")

pairs = []
for _, row in df.iterrows():
    ref = str(row["reference"])
    pairs.append((ref, str(row["hypA"])))
    pairs.append((ref, str(row["hypB"])))

# ==== RUN ====
results = []

for ref, hyp in pairs:
    print(f"Ref: {ref}\nHyp: {hyp}")
    label = classify(ref, hyp)
    print("→", label, "\n")

    results.append((ref, hyp, label))

# ==== SAVE ====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ref", "hyp", "class"])
    writer.writerows(results)

print("Done:", OUTPUT_CSV)