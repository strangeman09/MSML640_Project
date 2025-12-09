import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers, logging
import json
import os
import re
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

transformers.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "Salesforce/blip2-flan-t5-xl"   # OR "liuhaotian/llava-1.3-7b"

output_dir = f"./vqav2_embeddings_{model_name.split('/')[-1]}"
embedding_path = f"{output_dir}/vqav2_concat_embeddings.npy"
label_path = f"{output_dir}/vqav2_labels.npy"
model_save_path = f"{output_dir}/metric_model.pth"
prot_path = f"{output_dir}/prototypes.npy"
prot_ind_path = f"{output_dir}/prototype_indices.npy"


feat = np.load(embedding_path)
labels = np.load(label_path)
vec_len = feat.shape[1]

class DuelCNNWrapper(nn.Module):
    def __init__(self, vec_len):
        super().__init__()
        self.additional_layer = nn.Sequential(
            nn.Linear(vec_len, vec_len),
            nn.InstanceNorm1d(vec_len),
            nn.ReLU()
        )
    def forward(self, x):
        return self.additional_layer(x)

metric_model = DuelCNNWrapper(vec_len).to(device)
metric_model.load_state_dict(torch.load(model_save_path))
metric_model.eval()


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(feat, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=False)


new_feat = []
for batch, _ in tqdm(loader, desc="Embedding with metric model"):
    batch = batch.to(device)
    emb = metric_model(batch).detach().cpu().numpy()
    new_feat.append(emb)
new_feat = np.vstack(new_feat)


prototypes = np.load(prot_path)
prot_indices = np.load(prot_ind_path)


sup_idx, ref_idx = prot_indices[:2]


vqav2 = load_dataset("HuggingFaceM4/VQAv2", split="validation")

def load_img(path):
    from PIL import Image
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


def vqa_prompt(question, answer=None, few_shots=None):
    prompt = "Answer the visual question based on the image.\n\n"
    if few_shots:
        for q, a in few_shots:
            prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += f"Q: {question}\nA: "
    return prompt

few_shots = [
    (vqav2["question"][sup_idx], vqav2["multiple_choice_answer"][sup_idx]),
    (vqav2["question"][ref_idx], vqav2["multiple_choice_answer"][ref_idx]),
]


preds = []
y_true = []

for i in tqdm(range(len(vqav2)), desc="Evaluating VQAv2"):
    img = load_img(vqav2["image"][i])
    question = vqav2["question"][i]
    gt = vqav2["multiple_choice_answer"][i]

    prompt = vqa_prompt(question, few_shots=few_shots)
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=5)

    pred = processor.batch_decode(out, skip_special_tokens=True)[0]
    pred = pred.strip()

    preds.append(pred)
    y_true.append(gt)


def normalize(ans):
    ans = ans.lower()
    ans = re.sub(r"[^\w\s]", "", ans)
    return ans

y_true = [normalize(a) for a in y_true]
preds = [normalize(a) for a in preds]

accuracy = sum([t == p for t, p in zip(y_true, preds)]) / len(y_true)
print("Model:", model_name)
print("Prototype ICL Accuracy:", accuracy)

results = {
    "model": model_name,
    "accuracy": accuracy,
    "samples": len(y_true),
}

with open(f"{output_dir}/prototype_vqa_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Saved results to", f"{output_dir}/prototype_vqa_results.json")
