from tqdm import tqdm
# import torch.optim as optim
import torch
import warnings
import numpy as np
from losses import ManifoldPointToPointLoss,ProxyAnchorLoss
from manifold import grow_manifolds_supervised,calculate_point_point_similarities



warnings.filterwarnings("ignore", message="input's size at dim=0 does not match num_features")
def train_supcon_model(model, train_loader,proxy_anchor_loss,opt_mod,opt_prox, scheduler_mod,scheduler_prox, epochs=10, patience=5,momentum = 0.99,N_alpha = 4,N_beta = 0.5,delta = 2,m=3,reconstruction_threshold=0.9):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    momentum = momentum
    m = m
    N_alpha = N_alpha
    N_beta = N_beta
    delta = delta
    reconstruction_threshold = reconstruction_threshold
    
    

    loss1 = ManifoldPointToPointLoss(delta=delta)
    # model = model.to(device)
    best_loss = float('inf')
    best_epoch = 0
    epoch_losses = []  
    manifold_losses = []
    pca_losses = []
    early_stop_counter = 0  
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_sim_loss = 0.0
        total_metric_loss = 0.0
        model.train()

        for batch in train_loader:

            pos,lab = batch

            pos = pos.to(device).float()

            emb = model(pos)
            emb = emb.squeeze(1)
            lab= lab.to(device)
            manifold,basis,all_points = grow_manifolds_supervised(emb, lab, m=m, reconstruction_threshold=reconstruction_threshold, max_neighbors=20)
            sims = calculate_point_point_similarities(emb, manifold, basis, all_points, N_alpha=N_alpha, N_beta=N_beta)

            sim_loss = loss1(emb,sims)
            # emb = emb.squeeze(1)

            metric_loss = proxy_anchor_loss(emb,lab)
            loss = sim_loss + metric_loss
            total_loss += loss.item()
            total_sim_loss += sim_loss.item()
            total_metric_loss += metric_loss.item()

            # # Backward pass
            # # optimizer.zero_grad()
            # opt_mod.zero_grad()
            # opt_prox.zero_grad()
            # loss.backward()
            # # optimizer.step()
            # opt_mod.step()
            # opt_prox.step()
            opt_mod.zero_grad()
            loss.backward(retain_graph=True)  # Keep computation graph for second backward
            opt_mod.step()

            # For proxy update: Use only metric loss
            opt_prox.zero_grad()
            metric_loss.backward()
            opt_prox.step()
            proxy_anchor_loss.update_momentum_proxies(momentum)
            # update_momentum(model,momentum_model)

        # Track average loss per epoch
        avg_loss = total_loss / len(train_loader)
        

        avg_sim_loss = total_sim_loss/len(train_loader)
        avg_metric_loss = total_metric_loss/len(train_loader)
        
        epoch_losses.append(avg_loss)
        manifold_losses.append(avg_sim_loss)
        pca_losses.append(avg_metric_loss)
        # test_loss = evaluate_loader(test_loader,model)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.14f}, Best Loss: {best_loss:.14f},Sim loss:{avg_sim_loss:.14f},Metric loss:{avg_metric_loss:.14f}")

        # Check for improvement and apply early stopping if needed
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            # torch.save(model.state_dict(), "/export/kbodla/supcon_resnet.pth")
            # torch.save(model.state_dict(), "/export/kbodla/rl_model_last_layer_update_supconnew.pth") 
            early_stop_counter = 0  
        else:
            early_stop_counter += 1  
        if early_stop_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
            break
        scheduler_mod.step()
        scheduler_prox.step()
    return epoch_losses,manifold_losses,pca_losses,model


