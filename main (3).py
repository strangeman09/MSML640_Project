import numpy as np
from losses import ManifoldPointToPointLoss, ProxyAnchorLoss
from manifold import grow_manifolds_supervised, calculate_point_point_similarities
from train import train_supcon_model
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)



# Example:
# model_name = "Salesforce/blip2-flan-t5-xl"
# model_name = "liuhaotian/llava-1.3-7b"

model_name = "Salesforce/blip2-flan-t5-xl"

output_dir = f"./vqav2_embeddings_{model_name.split('/')[-1]}"
embedding_path = f"{output_dir}/vqav2_concat_embeddings.npy"
label_path = f"{output_dir}/vqav2_labels.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"


epochs = 200
lr = 0.001
refer_lab = [0, 1]   
momentum = 0.99
m = 3
N_beta = 0.5
N_alpha = 4
delta = 2
reconstruction_threshold = 0.9
alpha = 32
delta_pca = 0.1



feat = np.load(embedding_path)
lab = np.load(label_path)

feat_list = list(feat)
labels = list(lab)

vec_len = feat.shape[1]


prot_save_path = f"{output_dir}/prototypes.npy"
prot_close_ind_save_path = f"{output_dir}/prototype_indices.npy"
model_save_path = f"{output_dir}/metric_model.pth"

os.makedirs(output_dir, exist_ok=True)

proxy_anchor_loss = ProxyAnchorLoss(
    num_classes=len(refer_lab),
    embedding_dim=vec_len,
    alpha=alpha,
    delta_pca=delta_pca
).to(device)


class DuelCNNWrapper(nn.Module):
    def __init__(self, vec_len):
        super(DuelCNNWrapper, self).__init__()

        self.additional_layer = nn.Sequential(
            nn.Linear(vec_len, vec_len),
            nn.InstanceNorm1d(vec_len),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.additional_layer(x)
        return x

model = DuelCNNWrapper(vec_len).to(device)


class CustomDataset(Dataset):
    def __init__(self, img, labels):
        self.img = img
        self.labels = labels

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.labels[idx]

train_dataset = CustomDataset(feat_list, labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    drop_last=True
)


opt_mod = optim.Adam(model.parameters(), lr=lr)
opt_prox = optim.Adam(proxy_anchor_loss.parameters(), lr=lr)

scheduler_mod = torch.optim.lr_scheduler.ExponentialLR(opt_mod, gamma=0.97)
scheduler_prox = torch.optim.lr_scheduler.ExponentialLR(opt_prox, gamma=0.97)



classification_losses, manifold_losses, pca_losses, trained_model = train_supcon_model(
    model,
    train_loader,
    proxy_anchor_loss,
    opt_mod,
    opt_prox,
    scheduler_mod,
    scheduler_prox,
    epochs=epochs,
    momentum=momentum,
    N_alpha=N_alpha,
    N_beta=N_beta,
    delta=delta,
    m=m,
    reconstruction_threshold=reconstruction_threshold
)



proxies = proxy_anchor_loss.momentum_proxies.detach().cpu().numpy()
closest_vectors = []
closest_indices = []

for i, vec in enumerate(proxies):
    dists = np.linalg.norm(feat - vec, axis=1)
    closest_idx = np.argmin(dists)

    closest_vectors.append(feat[closest_idx])
    closest_indices.append(closest_idx)

closest_indices = np.array(closest_indices)
closest_vectors = np.array(closest_vectors)


np.save(prot_close_ind_save_path, closest_indices)
np.save(prot_save_path, proxies)
torch.save(trained_model.state_dict(), model_save_path)


