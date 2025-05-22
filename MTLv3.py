# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:03:46 2025

@author: Arpan Dam
"""

import pickle
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Load the preprocessed vectors
with open('influencer_vectors_pickle.pkl', 'rb') as f:
    vector_data = pickle.load(f)

# Load fairness scores
with open('fairness_scores_pickle.pkl', 'rb') as f:
    fairness_scores = pickle.load(f)
    

class InfluencerDataset(Dataset):
    def __init__(self, data_dict, fairness_scores):
        self.data = []
        for infl_id, v in data_dict.items():
            # Get fairness score from loaded data
            fairness_score = fairness_scores[infl_id]
            item = {
                'one_hot': torch.tensor(v['one_hot'], dtype=torch.float),
                'multi_hot': torch.tensor(v['multi_hot'], dtype=torch.float),
                'fairness': torch.tensor([fairness_score], dtype=torch.float)
            }
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]['one_hot'],
            self.data[idx]['multi_hot'],
            self.data[idx]['fairness']
        )


def simple_downsample_by_fairness(dataset):
    downsample_log = {}  # Dictionary to log before and after counts

    for idx, item in enumerate(dataset.data):
        fairness_score = item['fairness'].item()
        multi_hot = item['multi_hot']
        
        influenced_indices = [i for i, v in enumerate(multi_hot.tolist()) if v == 1]
        total_influenced = len(influenced_indices)

        downsample_log[idx] = {
            'before': total_influenced,
            'fairness': fairness_score
        }

        if total_influenced == 0:
            downsample_log[idx]['after'] = 0
            continue

        num_to_keep = int(fairness_score * total_influenced)
        indices_to_keep = random.sample(influenced_indices, num_to_keep)
        new_multi_hot = torch.zeros_like(multi_hot)
        for i in indices_to_keep:
            new_multi_hot[i] = 1

        item['multi_hot'] = new_multi_hot
        downsample_log[idx]['after'] = len(indices_to_keep)

    return downsample_log


def check_downsampling(dataset):
    for item in dataset.data:
        fairness_score = item['fairness'].item()
        multi_hot = item['multi_hot']

        original_influenced = torch.sum(multi_hot).item()
        print(f"Fairness: {fairness_score:.2f}, Original influenced: {original_influenced}")


class Fair2VecMTL(nn.Module):
    def __init__(self, input_dim, num_targets):
        super(Fair2VecMTL, self).__init__()
        self.linear = nn.Linear(input_dim, 50)
        self.influence_head = nn.Linear(50, num_targets)
        self.fairness_head = nn.Linear(50, 1)

    def forward(self, x):
        x_shared = F.relu(self.linear(x))
        influencer_embedding = F.relu(x_shared)
        pred_influence = torch.sigmoid(self.influence_head(influencer_embedding))
        pred_fairness = self.fairness_head(x_shared)
        return pred_influence, pred_fairness, influencer_embedding


def fair2vec_loss(pred_influence, true_influence, pred_fairness, true_fairness):
    bce_loss = F.binary_cross_entropy(pred_influence, true_influence)
    mse_loss = F.mse_loss(pred_fairness, true_fairness)
    return bce_loss + mse_loss


# Hyperparameters
num_users = len(vector_data[list(vector_data.keys())[0]]['one_hot'])
num_targets = len(vector_data[list(vector_data.keys())[0]]['multi_hot'])
batch_size = 100
num_epochs = 200
learning_rate = 0.001

# Dataset and loader
dataset = InfluencerDataset(vector_data, fairness_scores)
downsample_info = simple_downsample_by_fairness(dataset)

# Save influence count before and after downsampling
influencer_counts = {}
for idx, item in enumerate(dataset):
    infl_id = list(vector_data.keys())[idx]
    original_count = int(np.sum(vector_data[infl_id]['multi_hot']))
    downsampled_count = downsample_info[idx]['after']
    fairness = downsample_info[idx]['fairness']
    influencer_counts[infl_id] = {
        'original_count': original_count,
        'downsampled_count': downsampled_count,
        'fairness': fairness
    }

with open("influence_counts_before_after.pkl", "wb") as f:
    pickle.dump(influencer_counts, f)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = Fair2VecMTL(input_dim=num_users, num_targets=num_targets)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for one_hot_input, multi_hot_label, true_fairness in dataloader:
        pred_influence, pred_fairness, _ = model(one_hot_input)
        loss = fair2vec_loss(pred_influence, multi_hot_label, pred_fairness, true_fairness)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save influencer stats
influencer_stats = []
model.eval()
with torch.no_grad():
    for i, (infl_id, _) in enumerate(vector_data.items()):
        one_hot_tensor = dataset.data[i]['one_hot'].unsqueeze(0)
        _, _, embedding = model(one_hot_tensor)
        influencer_stats.append({
            'influencer_id': infl_id,
            'l1_norm': torch.norm(embedding, p=1).item(),
            'l2_norm': torch.norm(embedding, p=2).item(),
            'influence_count': int(dataset.data[i]['multi_hot'].sum().item()),
            'true_fairness': dataset.data[i]['fairness'].item(),
            'embedding_vector': embedding.squeeze(0).numpy()
        })

with open("influencer_stats.pkl", "wb") as f:
    pickle.dump(influencer_stats, f)

print("Training complete and results saved!")



import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load influencer stats
with open("influencer_stats.pkl", "rb") as f:
    influencer_stats = pickle.load(f)

# Extract data
embeddings = np.array([x['embedding_vector'] for x in influencer_stats])
influence_counts = np.array([x['influence_count'] for x in influencer_stats])
fairness_scores = np.array([x['true_fairness'] for x in influencer_stats])

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot 1: Colored by influence count
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=influence_counts, cmap='viridis', s=30)
plt.colorbar(label='Influence Count')
plt.title("t-SNE Plot of Influencer Embeddings (Colored by Influence Count)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_influence_count.png")
plt.show()

# Plot 2: Colored by fairness score
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=fairness_scores, cmap='plasma', s=30)
plt.colorbar(label='Fairness Score')
plt.title("t-SNE Plot of Influencer Embeddings (Colored by Fairness Score)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_fairness_score.png")
plt.show()