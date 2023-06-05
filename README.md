# Metric evaluator for Automatic Speech Recognition

## Context
Automatic tool to compute correlation between human perception of errors and metrics in the context of Automatic Speech Recognition (ASR).

Traditional metrics for asr such as Word Error Rate (WER) and Character Error Rate (CER) are often the subject of criticism from the speech community. Several researchers are trying to find alternatives for automatic system evaluation. To verify and compare these metrics seriously, it is necessary to study its correlation with human perception.

Using a french dataset named **HATS**, we have developed a tool to evaluate these metrics.

## Quickstart

Once the repository is downloaded, the evaluator need to call your metric.

```
def custom_metric(ref, hyp, memory=0):
    # compute a score given a textual reference and hypothesis
    # write your own metric here
    return score # lower-is-better rule

...
 
if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1
    cert_Y = 0.7

    x_score = evaluator(custom_metric, dataset, memory=memory, certitude=cert_X) # certitude is useful to filter utterances where humans are unsure
    y_score = evaluator(custom_metric, dataset, memory=memory, certitude=cert_Y)
```

## Note
The memory argument can be used to prevent loading a large model or data every time the function is called, as it can be loaded beforehand. 

## Cite

Please, cite the related paper if you use this framework.

```
@inproceedings{banerasroux,
  TITLE = {{HATS : Un jeu de donn{\'e}es int{\'e}grant la perception humaine appliqu{\'e}e {\`a} l'{\'e}valuation des m{\'e}triques de transcription de la parole}},
  AUTHOR = {Ba{\~n}eras-Roux, Thibault and Wottawa, Jane and Rouvier, Michael and Merlin, Teva and Dufour, Richard},
  URL = {https://hal.science/hal-04111840},
  BOOKTITLE = {{Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles}},
  ADDRESS = {Paris, France},
  ORGANIZATION = {{Association pour le Traitement Automatique des Langues (ATALA) and Association francophone de Recherche d'Information et Applications (ARIA)}},
  YEAR = {2023},
}
```
