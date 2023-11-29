import progressbar
import numpy
import os


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



def load_w2v_model():
    # HuggingFace model hub
    model_hub_w2v2 = "LeBenchmark/wav2vec2-FR-7K-large"
    model_w2v2 = HuggingFaceWav2Vec2(model_hub_w2v2, save_path='./save')
    return model_w2v2

def cosine_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))


def load_wav(path):
    waveform, sample_rate = torchaudio.load(path)
    # Resample if needed
    target_sample_rate = 16000  # Wav2Vec 2.0's expected sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    # Ensure mono-channel (1 channel)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def get_wav(text, tts): # save audio file if not exists
    text = text.lower()
    # check if files exist in dataset/audiofiles
    namefile = text
    # remove unexpected character
    accepted = "abcdefghijklmnopqrstuvwxyzéèêëàâäôöûüùîïç_"
    for x in namefile:
        if x not in accepted:
            namefile = namefile.replace(x, "_")
    # if audio file does not exists
    if not os.path.isfile("datasets/audiofiles/" + namefile + ".wav"):
        tts.tts_to_file(text, speaker_wav="datasets/audiofiles/bfm15.wav", language="fr", file_path="datasets/audiofiles/" + namefile + ".wav")
    # load audio file
    return load_wav("datasets/audiofiles/" + namefile + ".wav")

def get_features(wav, model_w2v2):
    features = model_w2v2.forward(wav)
    # compute only the average of the features in order to have the same shape
    average_features = torch.mean(features[0], dim=0, keepdim=True)[0]
    return average_features

def speech_difference(ref, hyp, memory):
    tts, model_w2v2 = memory
    # get audio files
    ref_wav = get_wav(ref, tts)
    hyp_wav = get_wav(hyp, tts)
    # get features
    ref_features = get_features(ref_wav, model_w2v2)
    hyp_features = get_features(hyp_wav, model_w2v2)
    # compute cosine similarity
    cs = cosine_similarity(ref_features, hyp_features)
    return (1-cs)*100 # lower is better



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


def write(namefile, x):
    with open("results/" + namefile + ".txt", "w", encoding="utf8") as file:
        file.write(namefile + "," + str(x) + "\n")



if __name__ == '__main__':
    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    cert_X = 1

    # useful for the metric but we do not need to recompute every time
    print("Importing...")


    import torch
    from TTS.api import TTS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts = TTS(model_path="../downstream/tts/model/tts_models--multilingual--multi-dataset--xtts_v2", config_path="../downstream/tts/model/tts_models--multilingual--multi-dataset--xtts_v2/config.json").to(device)

    from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
    import torchaudio

    model_w2v2 = load_w2v_model()

    result = evaluator(speech_difference, dataset, memory=(tts, model_w2v2), certitude=cert_X)
    write("wav2vec2", result)