import numpy as np
from losses import ManifoldPointToPointLoss,ProxyAnchorLoss
from manifold import grow_manifolds_supervised,calculate_point_point_similarities
from train import train_supcon_model
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os


torch.manual_seed(42) 

epochs = 50
lr = 0.00025
refer_lab = [0,1,2,3,4,5,6,7,8]
device = 'cuda'

parser = argparse.ArgumentParser(description='model name as input(hugging face id)')
parser.add_argument('--model_name', type=str, help='hugging face model name')
parser.add_argument('--device_id',type=int,nargs = '+',help='GPU ID',default = [0])
parser.add_argument('--momentum_constant',type=float,help='momentum constant for proxies update',default=0.99)
parser.add_argument('--manifold_m',type=int,help='manifold dims and n neighbors',default=3)
parser.add_argument('--N_beta',type=float,help='similarity calculation',default=0.5)
parser.add_argument('--N_alpha',type=float,help='similarity calculation',default=4)
parser.add_argument('--delta_manifold',type=float,help='similarity calculation',default=2)
parser.add_argument('--reconstruction_threshold',type=float,help='momentum constant for proxies update',default=0.9)
parser.add_argument('--alpha',type=int,help='alpha proxy anchor loss',default=32)
parser.add_argument('--delta_pca',type=float,help='delta for proxy anchor loss',default=0.1)

args = parser.parse_args()
model_name = args.model_name
gpu_id = args.device_id
momentum = args.momentum_constant
m = args.manifold_m
N_beta = args.N_beta
N_alpha = args.N_alpha
delta = args.delta_manifold
reconstruction_threshold = args.reconstruction_threshold
alpha = args.alpha
delta_pca = args.delta_pca



os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_id))

device = 'cuda'
model_name = args.model_name
model_data_path = model_name.split('/')[1]


direc_name = f''

new_direc_name = direc_name+f'/abilations'
os.makedirs(new_direc_name,exist_ok = True)

train_feat_path = direc_name+f'/encoding1_arr.npy'
train_action_path = direc_name+f'/encoding1_labels.npy'



feat = np.load(train_feat_path)
lab = np.load(train_action_path)

feat_img = list(feat)
labels=list(lab)

vec_len = feat.shape[1]

proxy_anchor_loss = ProxyAnchorLoss(num_classes=len(refer_lab), embedding_dim=vec_len,alpha=alpha,delta_pca=delta_pca).to(device)

######################################### path saving ###########################
def format_param_name(params: dict, precision: int = 3) -> str:
    """
    Format a dict of hyperparameters into a safe string for filenames.

    Args:
        params (dict): key-value pairs of parameters
        precision (int): decimal precision for floats (default=3)

    Returns:
        str: formatted string
    """
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            val_str = f"{value:.{precision}f}".replace(".", "_")
        else:
            val_str = str(value)
        parts.append(f"{key}{val_str}")
    return "_".join(parts)


params = {
    "momentum": momentum,
    "m": m,
    "N_beta": N_beta,
    "N_alpha": N_alpha,
    "delta": delta,
    "reconstruction_threshold": reconstruction_threshold,
    "alpha": alpha,
    "delta_pca": delta_pca
}

param_str = format_param_name(params)


prot_save_path = f"{direc_name}/abilations/prot_{param_str}.npy"
prot_close_ind_save_path = f"{direc_name}/abilations/prot_ind_{param_str}.npy"
model_save_path = f"{direc_name}/abilations/model_{param_str}.pth"
total_loss_path = f"{direc_name}/abilations/total_loss_{param_str}.npy"
manifold_loss_path = f"{direc_name}/abilations/manifold_loss_{param_str}.npy"
pca_loss_path = f"{direc_name}/abilations/pca_loss_{param_str}.npy"
############################### path saving##############################

######################## Model_def ####################
class DuelCNNWrapper(nn.Module):
    def __init__(self,vec_len):
        super(DuelCNNWrapper, self).__init__()
        
        self.additional_layer = nn.Sequential(
            nn.Linear(vec_len, vec_len),
            nn.InstanceNorm1d(vec_len),
            nn.ReLU()

           
        
        )

    def forward(self, x):
         
        x = self.additional_layer(x) 
        return x

model = DuelCNNWrapper(vec_len) 
model.to(device)
#################### Model_def #######################



#################### Dataset_def ####################
class CustomDataset(Dataset):
    def __init__(self, img, labels, transform=None):
        self.img = img
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        # Fetch the image and label corresponding to the index
        img = self.img[idx]
        label = self.labels[idx]
        
        # Apply the transformation if provided
        if self.transform:
            img = self.transform(img)
        
        return img, label

        

train_dataset = CustomDataset( feat_img, labels)
indices = list(range(len(train_dataset)))

indices_shuffled = torch.randperm(len(indices)).tolist()

shuffled_dataset = torch.utils.data.Subset(train_dataset, indices_shuffled)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False,num_workers = 4,drop_last = True)
###################### Dataset_def ###############################

opt_mod = torch.optim.Adam(model.parameters(), lr=0.001)  
opt_prox = torch.optim.Adam(proxy_anchor_loss.parameters(), lr=0.001)

scheduler_mod = torch.optim.lr_scheduler.ExponentialLR(opt_mod, gamma=0.97)
scheduler_prox = torch.optim.lr_scheduler.ExponentialLR(opt_prox, gamma=0.97)

classification_losses,manifold_losses,pca_losses,new_model = train_supcon_model(model, train_loader,proxy_anchor_loss,opt_mod,opt_prox, scheduler_mod,scheduler_prox, epochs=200,momentum = momentum,N_alpha = N_alpha,N_beta = N_beta,delta = delta,m=m,reconstruction_threshold= reconstruction_threshold)

##################### Prototype index ##############################
nn_human = proxy_anchor_loss.momentum_proxies.detach().cpu().numpy()
close_vec = []
close_ind = []

for i, vec in enumerate(nn_human):
    # Compute L2 distances between vec and all feat vectors
    dists = np.linalg.norm(feat - vec, axis=1)
    closest_idx = np.argmin(dists)
    closest_vec = feat[closest_idx]
    close_vec.append(closest_vec)
    close_ind.append(closest_idx)

close_ind=np.array(close_ind)
close_vec= np.array(close_vec) 
manifold_losses = np.array(manifold_losses)
total_losses = np.array(classification_losses)
pca_losses = np.array(pca_losses)

print(close_ind)
np.save(prot_close_ind_save_path,close_ind)
np.save(prot_save_path,nn_human)
torch.save(new_model.state_dict(),model_save_path)
np.save(total_loss_path,total_losses)
np.save(manifold_loss_path,manifold_losses)
np.save(pca_loss_path,pca_losses)
##################### Prototype index ##############################
