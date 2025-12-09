import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda'

class ManifoldPointToPointLoss(nn.Module):
    def __init__(self, delta=2.0):
       
        super(ManifoldPointToPointLoss, self).__init__()
        self.delta = delta
        
    def forward(self, embeddings, similarities):
       
        batch_size = embeddings.size(0)
   
        distances = torch.cdist(embeddings, embeddings, p=2)
        
       
        target_distances = self.delta * (1.0 - similarities)
        
    
        squared_errors = (distances - target_distances)**2
        
      
        mask = 1.0 - torch.eye(batch_size, device=embeddings.device)
        squared_errors = squared_errors * mask
        
    
        num_pairs = batch_size * (batch_size - 1)
        loss = torch.sum(squared_errors) / num_pairs
        
        return loss




class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, alpha=32, delta_pca=0.1):
        super(ProxyAnchorLoss, self).__init__()
        self.alpha = alpha
        self.delta_pca = delta_pca
        
      
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    
        self.momentum_proxies = nn.Parameter(self.proxies.clone().detach(), requires_grad=False)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.momentum_proxies, p=2, dim=1)

        similarity = embeddings @ proxies.T  # [batch_size, num_classes]

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=proxies.shape[0]).float()
        pos_mask = labels_one_hot.bool()
        neg_mask = ~pos_mask

        pos_exp = torch.exp(-self.alpha * (similarity - self.delta_pca)) * pos_mask
        neg_exp = torch.exp(self.alpha * (similarity - self.delta_pca)) * neg_mask

        pos_term = torch.log(1 + pos_exp.sum(dim=1)).mean()
        neg_term = torch.log(1 + neg_exp.sum(dim=1)).mean()

        loss = pos_term + neg_term
        return loss

    @torch.no_grad()
    def update_momentum_proxies(self, momentum=0.99):
        
        self.momentum_proxies.data = momentum * self.momentum_proxies.data + (1 - momentum) * self.proxies.data


# proxy_anchor_loss = ProxyAnchorLoss(num_classes=len(refer_lab), embedding_dim=1152).to(device)