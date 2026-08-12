import progressbar
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def load_hf_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Before anything, don't forget to run: export TRANSFORMERS_OFFLINE=1")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     offload_buffers=True,
    #     trust_remote_code=True,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     offload_buffers=True,
    #     trust_remote_code=True,
    #     ignore_mismatched_sizes=True,
    #     quantization_config=None,   # force-ignore the embedded FP8 config
    # )
    # exit()
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, ref, hypA, hypB, shot=True, cot=True):
    if cot:
        SYSTEM = (
            "Une référence est une transcription exacte d'un audio. "
            "Deux hypothèses fausses sont proposées.\n"
            "Explique ta réflexion et finis ta phrase en écrivant "
            "'A', 'B' ou 'a' ou 'b' si indécis."
        )
    else:
        SYSTEM = (
            "Une référence est une transcription exacte d'un audio. "
            "Deux hypothèses fausses sont proposées.\n"
            "Choisis 'A' ou 'B' selon laquelle est la plus acceptable."
        )
    user_content = (
        f"Référence : {ref}\n"
        f"Hypothèse A : {hypA}\n"
        f"Hypothèse B : {hypB}\n\n"
    )
    shot_ref = "c' est à lui même"
    shot_hypA = "êtes à lui même"
    shot_hypB = "c' est euh à lui-même"
    if cot:
        shot_cot = "Même si l'hypothèse B contient une disfluence ('euh'), elle correspond beaucoup mieux à la référence en termes de mots et de sens. La disfluence peut être tolérée si elle fait partie de l'original, tandis que l'erreur grammaticale de l'hypothèse A est plus problématique. Donc, la transcription la plus acceptable est l'hypothèse B."
    else:
        shot_cot = "B"

    shot_user_content = (
        f"Référence : {shot_ref}\n"
        f"Hypothèse A : {shot_hypA}\n"
        f"Hypothèse B : {shot_hypB}\n\n"
    )

    if tokenizer.chat_template:
        if shot:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": f"{shot_user_content}"},
                {"role": "assistant", "content": shot_cot},
                {"role": "user",   "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user_content},
            ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=(not cot),
        )
    else:
        return f"{SYSTEM}\n\n{user_content}\nRéponse :"


def hf_predict_batch(tokenizer, model, batch_items, cot, max_new_tokens=4096, verbose=False):
    """
    batch_items: list of dicts with keys 'reference', 'hypA', 'hypB'
    Returns a list of 'A', 'B', or None for each item.
    """
    prompts = [
        build_prompt(tokenizer, item["reference"], item["hypA"], item["hypB"], shot=True, cot=cot)
        for item in batch_items
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    # prompt_lengths = inputs["input_ids"].shape[1]
    # for output in output_ids:
    for i, output in enumerate(output_ids):
        prompt_lengths = inputs["input_ids"][i].shape[0]    
        new_tokens = output[prompt_lengths:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if verbose:
            print(f"reference: {batch_items[results.__len__()]['reference']}")
            print(f"hypA     : {batch_items[results.__len__()]['hypA']}")
            print(f"hypB     : {batch_items[results.__len__()]['hypB']}")
            print(f"Model reply: {reply}")
            input("Press Enter to continue...")
        matches = re.findall(r'[AaBb]', reply)
        results.append(matches[-1].upper() if matches else None)

    return results


def hf_pairwise_evaluator(dataset, tokenizer, model, certitude=0.3, batch_size=8, cot=True, verbose=True):
    ignored = 0
    accepted = 0
    correct = 0
    incorrect = 0

    # Pre-filter accepted items to batch efficiently
    accepted_items = []
    ignored_count_prefilter = 0
    for item in dataset:
        nbrA, nbrB = item["nbrA"], item["nbrB"]
        if nbrA + nbrB < 5:
            ignored_count_prefilter += 1
            continue
        c = max(nbrA, nbrB) / (nbrA + nbrB)
        if c >= certitude:
            accepted_items.append(item)
        else:
            ignored_count_prefilter += 1

    ignored += ignored_count_prefilter

    if verbose:
        bar = progressbar.ProgressBar(max_value=len(accepted_items))

    # Process in batches
    for batch_start in range(0, len(accepted_items), batch_size):
        batch = accepted_items[batch_start: batch_start + batch_size]
        predictions = hf_predict_batch(tokenizer, model, batch, cot=cot)

        for item, chosen in zip(batch, predictions):
            nbrA, nbrB = item["nbrA"], item["nbrB"]
            if chosen is None:
                ignored += 1
                continue

            accepted += 1
            if (chosen == 'A' and nbrA > nbrB) or (chosen == 'B' and nbrB > nbrA):
                correct += 1
            else:
                incorrect += 1

        if verbose:
            bar.update(min(batch_start + batch_size, len(accepted_items)))

    print()
    if correct + incorrect == 0:
        print("No valid comparisons made.")
        return 0

    ratio = correct / (correct + incorrect) * 100
    print(f"ratio correct : {ratio:.2f}%")
    print(f"ratio ignored : {ignored / (ignored + accepted) * 100:.2f}%")
    return ratio


def read_dataset(dataname):
    dataset = []
    with open("datasets/" + dataname, "r", encoding="utf8") as file:
        next(file)
        for line in file:
            line = line[:-1].split("\t")
            dataset.append({
                "reference": line[0],
                "hypA":      line[1],
                "nbrA":      int(line[2]),
                "hypB":      line[3],
                "nbrB":      int(line[4]),
            })
    return dataset


if __name__ == "__main__":
    # Swap model_name for anything instruction-tuned on HuggingFace
    # Faster/smaller : "Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it"
    # More accurate  : "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3-8B-Instruct"
    # MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct" # "Qwen/Qwen2.5-1.5B-Instruct"
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1"
    # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # MODEL_NAME = "Qwen/Qwen3-8B"
    # MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_NAME = "google/gemma-4-31B"
    # MODEL_NAME = 
    # MODEL_NAME = 
    # MODEL_NAME = 
    BATCH_SIZE = 8   # increase if you have more VRAM (16, 32...)
    cot = False

    tokenizer, model = load_hf_model(MODEL_NAME)

    print("Reading dataset...")
    dataset = read_dataset("hats.txt")

    # 49.87%
    print("\n--- Evaluating with certitude=1.0 ---")
    x_score = hf_pairwise_evaluator(dataset, tokenizer, model, certitude=1.0, batch_size=BATCH_SIZE, cot=cot)

    print("\n--- Evaluating with certitude=0.7 ---")
    y_score = hf_pairwise_evaluator(dataset, tokenizer, model, certitude=0.7, batch_size=BATCH_SIZE, cot=cot)

    print("\n--- Evaluating with certitude=0.0 ---")
    z_score = hf_pairwise_evaluator(dataset, tokenizer, model, certitude=0.0, batch_size=BATCH_SIZE, cot=cot)

    # save results to a file
    with open("out/results.txt", "a") as f:
        f.write(f"{MODEL_NAME}\t{BATCH_SIZE}\t{x_score:.2f}\t{y_score:.2f}\t{z_score:.2f}\n")