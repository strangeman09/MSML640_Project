import torch
import numpy as np
from sklearn.decomposition import PCA

def grow_manifolds_supervised(batch_embeddings, batch_labels, m=3, reconstruction_threshold=0.9, max_neighbors=20):

    embeddings = batch_embeddings.detach().cpu().numpy()
    labels = batch_labels.detach().cpu().numpy()
  
    unique_classes = np.unique(labels)
    
    manifolds = {}  
    manifold_bases = {}  
    
   
    all_manifold_points = set()
    
    for cls in unique_classes:
       
        class_indices = np.where(labels == cls)[0]
        
        if len(class_indices) < m:
            print(f"Skipping class {cls} - not enough points ({len(class_indices)} < {m})")
            continue
            
        
        anchor_idx = class_indices[0]
        
       
        manifold_points = [anchor_idx]
        
     
        remaining_indices = class_indices[1:]
        
       
        distances = np.array([
            np.linalg.norm(embeddings[anchor_idx] - embeddings[idx]) 
            for idx in remaining_indices
        ])
        
        
        sorted_indices = remaining_indices[np.argsort(distances)]
        
       
        initial_neighbors = sorted_indices[:m-1]
        manifold_points.extend(initial_neighbors)
        
      
        X_i = embeddings[manifold_points]
        
       
        pca = PCA(n_components=m)
        # print(X_i.shape)
        pca.fit(X_i)
        
       
        candidates = [idx for idx in sorted_indices[m-1:] if idx not in manifold_points]
        candidates = candidates[:max_neighbors]  
        
        
        for candidate_idx in candidates:
            candidate = embeddings[candidate_idx]
            
            
            projected = pca.transform(candidate.reshape(1, -1))
            reconstructed = pca.inverse_transform(projected)
            
          
            error = np.linalg.norm(candidate - reconstructed.flatten()) / np.linalg.norm(candidate)
            
           
            if error < (1 - reconstruction_threshold):
                manifold_points.append(candidate_idx)
                
               
                X_i = embeddings[manifold_points]
                pca = PCA(n_components=m)
                pca.fit(X_i)
        
        
        manifolds[cls] = manifold_points
        
        
        manifold_bases[cls] = pca.components_
        
       
        all_manifold_points.update(manifold_points)
    
    
    all_indices = set(range(len(embeddings)))
    non_manifold_points = list(all_indices - all_manifold_points)
    
    return manifolds, manifold_bases, non_manifold_points



def grow_manifolds_unsupervised(batch_embeddings, batch_labels, n_anchors=None, m=3, reconstruction_threshold=0.9, max_neighbors=20):

    # Convert tensors to numpy for easier manipulation
    embeddings = batch_embeddings.detach().cpu().numpy()
    labels = batch_labels.detach().cpu().numpy()
    
    batch_size = embeddings.shape[0]
    
    
    if n_anchors is None:
        n_anchors = len(np.unique(labels))
    
    
    anchor_indices = np.random.choice(batch_size, size=n_anchors, replace=False)
    
    manifolds = {}  
    manifold_bases = {}  
    

    all_manifold_points = set()
    
    for anchor_idx in anchor_indices:
       
        manifold_points = [anchor_idx]
        
       
        distances = np.array([
            np.linalg.norm(embeddings[anchor_idx] - embeddings[idx]) 
            if idx != anchor_idx else float('inf') 
            for idx in range(batch_size)
        ])
        
       
        sorted_indices = np.argsort(distances)
        
       
        initial_neighbors = sorted_indices[:m-1]
        manifold_points.extend(initial_neighbors)
        
       
        X_i = embeddings[manifold_points]
        
        
        pca = PCA(n_components=m)
        pca.fit(X_i)
        
      
        candidates = [idx for idx in sorted_indices[m-1:] if idx not in manifold_points]
        candidates = candidates[:max_neighbors] 
        
      
        for candidate_idx in candidates:
            candidate = embeddings[candidate_idx]
            
           
            projected = pca.transform(candidate.reshape(1, -1))
            reconstructed = pca.inverse_transform(projected)
            
           
            error = np.linalg.norm(candidate - reconstructed.flatten()) / np.linalg.norm(candidate)
            
          
            if error < (1 - reconstruction_threshold):
                manifold_points.append(candidate_idx)
                
               
                X_i = embeddings[manifold_points]
                pca = PCA(n_components=m)
                pca.fit(X_i)
        
        
        manifolds[anchor_idx] = manifold_points
        
        
        manifold_bases[anchor_idx] = pca.components_
        
     
        all_manifold_points.update(manifold_points)
    

    all_indices = set(range(batch_size))
    non_manifold_points = list(all_indices - all_manifold_points)
    
    return manifolds, manifold_bases, non_manifold_points

import torch
import numpy as np

def calculate_point_point_similarities(batch_embeddings, manifolds, manifold_bases, non_manifold_points, N_alpha=4, N_beta=0.5):

    embeddings = batch_embeddings.detach().cpu().numpy()
    batch_size = embeddings.shape[0]
    

    similarities = np.zeros((batch_size, batch_size))
    
 
    point_to_manifold = {}
    for cls, points in manifolds.items():
        for point_idx in points:
            point_to_manifold[point_idx] = cls
  
    def calculate_one_way_similarity(point_i_idx, point_j_idx, j_manifold_label):
       
        point_i = embeddings[point_i_idx]
        point_j = embeddings[point_j_idx]
        
     
        basis = manifold_bases[j_manifold_label]
        
       
        projection_coeffs = np.dot(point_i - point_j, basis.T)
        projection = point_j + np.dot(projection_coeffs, basis)
        
        
        orthogonal_vector = point_i - projection
        orthogonal_distance = np.linalg.norm(orthogonal_vector)
        
     
        projected_distance = np.linalg.norm(projection - point_j)
        
      
        alpha = 1 / ((1 + orthogonal_distance**2)**N_alpha)
        
      
        beta = 1 / ((1 + projected_distance)**N_beta)
        
      
        return alpha * beta
    
   
    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
               
                similarities[i, j] = 1.0
                continue
                
            
            similarity_i_to_j = 0.0
            similarity_j_to_i = 0.0
            
            
            if i in point_to_manifold:
               
                if j in point_to_manifold:
                    
                    i_manifold = point_to_manifold[i]
                    j_manifold = point_to_manifold[j]
                    
                   
                    similarity_i_to_j = calculate_one_way_similarity(i, j, j_manifold)
                    
                   
                    similarity_j_to_i = calculate_one_way_similarity(j, i, i_manifold)
                else:
                   
                    i_manifold = point_to_manifold[i]
                    similarity_j_to_i = calculate_one_way_similarity(j, i, i_manifold)
                    similarity_i_to_j = 0  
            else:
               
                if j in point_to_manifold:
                    
                    j_manifold = point_to_manifold[j]
                    similarity_i_to_j = calculate_one_way_similarity(i, j, j_manifold)
                else:
                  
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                  
                    similarity_i_to_j = np.exp(-dist)
                    similarity_j_to_i = similarity_i_to_j
            
            
            similarities[i, j] = (similarity_i_to_j + similarity_j_to_i) / 2.0
    
    return torch.tensor(similarities, device=batch_embeddings.device)