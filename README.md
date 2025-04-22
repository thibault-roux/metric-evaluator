# Metric evaluator for Automatic Speech Recognition

Metric-Evaluator is a easy-to-use French 🇫🇷 toolkit to evaluate your own metric for Automatic Speech Recognition (ASR) using the HATS data set.

## 🔍 Motivation

Traditional ASR metrics like Word Error Rate (WER) and Character Error Rate (CER) often face critics within the speech community 👩🏽‍🔬 Recognizing the need for alternative evaluation methods, several researchers are exploring new methods. To genuinely validate and compare these metrics, it's imperative to assess their correlation with human perception 🧠

## 🗃️ HATS Dataset

**HATS** (**H**uman **A**ssessed **T**ranscription **S**ide-by-Side) is a data set for French 🇫🇷 which consists of 1,000 triplets (reference, hypothesis A, hypothesis B) and 7,150 human choice annotated by 143 subjects 🫂 Their objective was to select, given a textual reference, which of two erroneous hypotheses is the best.

## 🧑‍🏫 Metric-Evaluator

This toolkit calculates the percentage of time a metric agrees with human judgments. Recognizing that human judgments can vary, instances arise where no consensus exists, and choices may be influenced by randomness 🎲 To filter the dataset based on consensus cases, utilize the certitude argument. This parameter represents the percentage of humans who selected the same hypothesis (set it to 1 when 100% of subjects make the same choice, and 0.7 when 70% of subjects choose the same hypothesis)."

## 🚀 Quickstart

- Step 1: ```git clone https://github.com/thibault-roux/metric-evaluator.git```
- Step 2: Write your own custom metric using the memory argument to avoid reloading a model if needed.
- Step 3: Evaluate your metric with the certitude you want!

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

## 📊 Results

| Metrics                          | 100%      | 70%       | Full      |
| -------------------------------- | --------- | --------- | --------- |
| Word Error Rate                  | 63%       | 53%       | 49%       |
| Character Error Rate             | 77%       | 64%       | 60%       |
| BERTScore CamemBERT-large        | 80%       | 68%       | 65%       |
| SemDist CamemBERT-large          | 80%       | 71%       | 67%       |
| SemDist Sentence CamemBERT-large | **90%**   | **78%**   | **73%**   |
| Phoneme Error Rate               | 80%       | 69%       | 64%       |

To add the results of your metric, contact me at **thibault [le dot] roux [le at] uclouvain.be** ✉️

## 📜 Cite

Please, cite the related paper if you use this toolkit or the HATS dataset.

```
@inproceedings{baneras2023hats,
  title={HATS: An Open data set Integrating Human Perception Applied to the Evaluation of Automatic Speech Recognition Metrics},
  author={Ba{\~n}eras-Roux, Thibault and Wottawa, Jane and Rouvier, Mickael and Merlin, Teva and Dufour, Richard},
  booktitle={Text, Speech and Dialogue 2023 - Interspeech Satellite},
  year={2023}
}
```
