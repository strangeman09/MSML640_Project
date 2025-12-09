from tqdm import tqdm
import json
import pandas as pd
import os
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from datasets import load_dataset
from PIL import Image


model_name = "Salesforce/blip2-flan-t5-xl"  # OR "liuhaotian/llava-1.3-7b"
output_dir = f"./vqav2_embeddings_{model_name.split('/')[-1]}"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


if "blip2" in model_name.lower():
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", output_hidden_states=True
    )
elif "llava" in model_name.lower():
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", output_hidden_states=True
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


ds = load_dataset("HuggingFaceM4/VQAv2")["train"]

def load_img(img):
    return Image.open(img).convert("RGB")


def embed_batch_text(texts, max_length=128):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]

    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy()

def embed_batch_images(images):
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.vision_model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]
    pooled = hidden.mean(dim=1)  # avg pooling for images
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()


batch_size = 8
img_embeds = []
txt_embeds = []
labels = []

for i in tqdm(range(0, len(ds), batch_size), desc="Processing VQAv2"):
    batch = ds[i:i+batch_size]

    batch_imgs = [load_img(path) for path in batch["image"]]
    batch_txt = [
        f"Question: {q} Answer: {a}"
        for q, a in zip(batch["question"], batch["multiple_choice_answer"])
    ]

    t_emb = embed_batch_text(batch_txt)
    i_emb = embed_batch_images(batch_imgs)

    txt_embeds.append(t_emb)
    img_embeds.append(i_emb)
    labels.extend(batch["multiple_choice_answer"])

torch.cuda.empty_cache()


txt_arr = np.vstack(txt_embeds)
img_arr = np.vstack(img_embeds)

concat_arr = np.concatenate([txt_arr, img_arr], axis=1)
labels = np.array(labels)

np.save(f"{output_dir}/vqav2_concat_embeddings.npy", concat_arr)
np.save(f"{output_dir}/vqav2_labels.npy", labels)

print("Final embedding shape:", concat_arr.shape)
print("Labels shape:", labels.shape)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

flush()

