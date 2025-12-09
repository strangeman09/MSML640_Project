import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
import transformers
import json
import os
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader

transformers.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "Salesforce/blip2-flan-t5-xl"  # OR: "liuhaotian/llava-1.3-7b"

output_dir = f"./vqav2_embeddings_{model_name.split('/')[-1]}"
embedding_path = f"{output_dir}/vqav2_concat_embeddings.npy"
label_path = f"{output_dir}/vqav2_labels.npy"


vqav2 = load_dataset("HuggingFaceM4/VQAv2", split="validation")

def load_img(path):
    return Image.open(path).convert("RGB")


if "blip2" in model_name.lower():
    processor = AutoProcessor.from_pretrained(model_name)
    vlm = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
else:
    processor = AutoProcessor.from_pretrained(model_name)
    vlm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )


def vqa_prompt(question, few_shots=None):
    prompt = "Answer the visual question based on the image.\n\n"
    
    if few_shots:
        for q, a in few_shots:
            prompt += f"Q: {q}\nA: {a}\n\n"

    prompt += f"Q: {question}\nA: "
    return prompt


def normalize(ans):
    ans = ans.lower()
    ans = re.sub(r"[^\w\s]", "", ans)
    return ans.strip()


def zero_shot_eval():
    preds = []
    truths = []
    
    for i in tqdm(range(len(vqav2)), desc="Zero-Shot Baseline:"):
        img = load_img(vqav2["image"][i])
        question = vqav2["question"][i]
        gt = vqav2["multiple_choice_answer"][i]

        prompt = vqa_prompt(question)
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=5)

        pred = processor.batch_decode(out, skip_special_tokens=True)[0]
        preds.append(normalize(pred))
        truths.append(normalize(gt))
    
    accuracy = sum([p == t for p, t in zip(preds, truths)]) / len(preds)
    return accuracy


feat = np.load(embedding_path)
labels = np.load(label_path)

def find_nearest_neighbor(index):
   
    similarities = np.dot(feat, feat[index]) / (
        np.linalg.norm(feat, axis=1) * np.linalg.norm(feat[index])
    )
  
    sorted_idx = np.argsort(similarities)[::-1]
    if sorted_idx[0] == index:
        return sorted_idx[1]
    else:
        return sorted_idx[0]

def similarity_based_one_shot_eval():
    preds = []
    truths = []

    for i in tqdm(range(len(vqav2)), desc="Similarity-based One-Shot:"):
        img = load_img(vqav2["image"][i])
        question = vqav2["question"][i]
        gt = vqav2["multiple_choice_answer"][i]

        # Retrieve nearest-neighbor sample as few-shot
        nn_idx = find_nearest_neighbor(i)
        fs_question = vqav2["question"][nn_idx]
        fs_answer = vqav2["multiple_choice_answer"][nn_idx]
        few_shots = [(fs_question, fs_answer)]

        prompt = vqa_prompt(question, few_shots)

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=5)

        pred = processor.batch_decode(out, skip_special_tokens=True)[0]
        preds.append(normalize(pred))
        truths.append(normalize(gt))

    accuracy = sum([p == t for p, t in zip(preds, truths)]) / len(preds)
    return accuracy


print("\nRunning Zero-Shot Evaluation...")
zs_accuracy = zero_shot_eval()
print(f"Zero-Shot Accuracy: {zs_accuracy:.4f}")

print("\nRunning Similarity-Based One-Shot Evaluation...")
os_accuracy = similarity_based_one_shot_eval()
print(f"One-Shot Similarity Accuracy: {os_accuracy:.4f}")

results = {
    "model": model_name,
    "zero_shot_accuracy": zs_accuracy,
    "one_shot_similarity_accuracy": os_accuracy,
}

save_path = f"{output_dir}/baseline_vqa_results.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved results →", save_path)
