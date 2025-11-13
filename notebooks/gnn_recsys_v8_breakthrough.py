"""
GNN-based Recommendation System - V8: BREAKTHROUGH VERSION
==========================================================

Triple Boost Strategy:
1. User/Item Bias Modeling - 개인별 선호도 편향 명시적 모델링
2. Multi-Task Learning - Rating Regression + BPR Ranking 동시 학습
3. Hard Negative Sampling - Low-rating 아이템을 negative로 사용

Expected improvement: +5~10% absolute gain over baseline
"""

import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("V8 BREAKTHROUGH - Triple Boost Strategy")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data split
    'train_ratio': 0.7,
    'valid_ratio': 0.1,
    'test_ratio': 0.2,

    # Model architecture
    'embedding_dim': 64,
    'n_layers': 2,

    # Training parameters
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 100,
    'patience': 20,  # More patience for complex model

    # ⭐ V8 NEW: Multi-task loss weights
    'alpha': 0.3,  # Rating regression weight (30%)
    'beta': 0.7,   # BPR ranking weight (70%)

    # ⭐ V8 NEW: Negative sampling
    'neg_ratio': 4,
    'hard_neg_ratio': 0.5,  # 50% hard negative, 50% random

    # ⭐ V8 NEW: Graph augmentation
    'edge_dropout': 0.1,  # 10% edge dropout during training

    # Rating thresholds
    'low_rating_threshold': 3.0,   # < 3.0 for hard negatives
    'high_rating_threshold': 3.5,  # >= 3.5 for positives

    # Evaluation
    'top_k': 10,

    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,

    # Paths
    'data_dir': '../data',
    'processed_dir': '../data/processed',
    'model_dir': '../models',
    'result_dir': '../results',
}

# Create directories
for dir_path in [CONFIG['processed_dir'], CONFIG['model_dir'], CONFIG['result_dir']]:
    os.makedirs(dir_path, exist_ok=True)

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(CONFIG['seed'])

print("\n⭐ V8 Configuration:")
print(f"  - Embedding dim: {CONFIG['embedding_dim']}")
print(f"  - Layers: {CONFIG['n_layers']}")
print(f"  - Learning rate: {CONFIG['learning_rate']}")
print(f"  - Batch size: {CONFIG['batch_size']}")
print(f"  - Alpha (rating weight): {CONFIG['alpha']}")
print(f"  - Beta (ranking weight): {CONFIG['beta']}")
print(f"  - Hard negative ratio: {CONFIG['hard_neg_ratio']}")
print(f"  - Edge dropout: {CONFIG['edge_dropout']}")
print(f"  - Device: {CONFIG['device']}")

# ============================================================================
# Data Loading
# ============================================================================

print("\n" + "=" * 80)
print("Loading Data")
print("=" * 80)

# Load ID mappings
with open(os.path.join(CONFIG['processed_dir'], 'id_mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)

n_users = len(mappings['user_id_map'])
n_items = len(mappings['item_id_map'])

print(f"Users: {n_users}")
print(f"Items: {n_items}")

# Load splits
train_df = pd.read_csv(os.path.join(CONFIG['processed_dir'], 'train_split_v3.csv'))
valid_df = pd.read_csv(os.path.join(CONFIG['processed_dir'], 'valid_split_v3.csv'))
test_df = pd.read_csv(os.path.join(CONFIG['processed_dir'], 'test_split_v3.csv'))

print(f"\nTrain: {len(train_df):,}")
print(f"Valid: {len(valid_df):,}")
print(f"Test:  {len(test_df):,}")

# ⭐ V8 NEW: Prepare low-rating data for hard negatives
print("\n⭐ Preparing hard negative data (rating < {:.1f})".format(CONFIG['low_rating_threshold']))

# Load original data to get low ratings
train_all_ratings = pd.read_csv(os.path.join(CONFIG['processed_dir'], 'train_split.csv'))
low_rating_df = train_all_ratings[train_all_ratings['rating'] < CONFIG['low_rating_threshold']]

print(f"Low-rating samples: {len(low_rating_df):,}")
print(f"High-rating samples (train): {len(train_df):,}")
print(f"Low/High ratio: {len(low_rating_df)/len(train_df):.2f}")

# Create user -> low-rating items dict
low_rating_items_dict = defaultdict(set)
for _, row in low_rating_df.iterrows():
    low_rating_items_dict[row['user_id']].add(row['item_id'])

print(f"Users with low ratings: {len(low_rating_items_dict)}")

# ============================================================================
# Graph Construction
# ============================================================================

print("\n" + "=" * 80)
print("Graph Construction")
print("=" * 80)

def create_graph(df, n_users, n_items):
    """Create bipartite graph"""
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values + n_users

    edge_index = torch.tensor([
        np.concatenate([user_ids, item_ids]),
        np.concatenate([item_ids, user_ids])
    ], dtype=torch.long)

    print(f"  Nodes: {n_users + n_items} (Users: {n_users}, Items: {n_items})")
    print(f"  Edges: {edge_index.shape[1]:,} (bidirectional)")

    return edge_index

print("\n[Train Graph]")
train_edge_index = create_graph(train_df, n_users, n_items)

print("\n[Train+Valid Graph]")
train_valid_df = pd.concat([train_df, valid_df])
train_valid_edge_index = create_graph(train_valid_df, n_users, n_items)

# ============================================================================
# User-Item Dictionaries
# ============================================================================

def create_user_item_dict(df):
    """User -> set of items"""
    user_items = defaultdict(set)
    for _, row in df.iterrows():
        user_items[row['user_id']].add(row['item_id'])
    return user_items

train_user_items = create_user_item_dict(train_df)
train_valid_user_items = create_user_item_dict(train_valid_df)

print("\n✅ Data preparation complete!")

# ============================================================================
# V8 NEW: Hard Negative Sampling
# ============================================================================

def hard_negative_sampling(batch_df, user_items_dict, n_items,
                           low_rating_items_dict, neg_ratio=1, hard_ratio=0.5):
    """
    Hard negative sampling strategy

    Args:
        batch_df: Batch dataframe with positive samples
        user_items_dict: User -> positive items mapping
        n_items: Total number of items
        low_rating_items_dict: User -> low-rating items mapping
        neg_ratio: Number of negatives per positive
        hard_ratio: Ratio of hard negatives (0.0 ~ 1.0)

    Returns:
        neg_users, neg_items: Arrays of negative samples
    """
    pos_users = batch_df['user_id'].values
    pos_items = batch_df['item_id'].values

    neg_users = []
    neg_items = []

    for user_id, pos_item in zip(pos_users, pos_items):
        user_pos_items = user_items_dict[user_id]
        user_low_items = low_rating_items_dict.get(user_id, set())

        for _ in range(neg_ratio):
            # Hard negative: sample from user's low-rating items
            if random.random() < hard_ratio and len(user_low_items) > 0:
                # Filter out already sampled positives
                available_low = user_low_items - user_pos_items
                if len(available_low) > 0:
                    neg_item = random.choice(list(available_low))
                else:
                    # Fallback to random
                    while True:
                        neg_item = random.randint(0, n_items - 1)
                        if neg_item not in user_pos_items:
                            break
            # Random negative
            else:
                while True:
                    neg_item = random.randint(0, n_items - 1)
                    if neg_item not in user_pos_items:
                        break

            neg_users.append(user_id)
            neg_items.append(neg_item)

    return np.array(neg_users), np.array(neg_items)

print("\n⭐ Hard negative sampling function defined!")

# ============================================================================
# V8 NEW: Edge Dropout for Graph Augmentation
# ============================================================================

def apply_edge_dropout(edge_index, dropout_rate=0.1):
    """
    Randomly drop edges for graph augmentation

    Args:
        edge_index: [2, num_edges]
        dropout_rate: Probability of dropping an edge

    Returns:
        Augmented edge_index
    """
    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges) > dropout_rate
    return edge_index[:, mask]

print("⭐ Edge dropout function defined!")

# ============================================================================
# V8 Model Architecture
# ============================================================================

class LightGCNConv(MessagePassing):
    """LightGCN Convolution Layer"""
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN_V8(nn.Module):
    """
    LightGCN V8: BREAKTHROUGH Version

    Features:
    1. User/Item Bias terms
    2. Multi-task: Rating regression + BPR ranking
    3. Edge dropout for augmentation
    """

    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # ⭐ BOOST #1: Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)

        # Graph convolution layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])

        # ⭐ BOOST #2: Rating regression MLP
        self.rating_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, edge_index, edge_dropout=0.0):
        """
        Graph forward pass

        Args:
            edge_index: Edge connectivity
            edge_dropout: Edge dropout rate (for augmentation during training)

        Returns:
            user_emb, item_emb: Final embeddings after graph convolution
        """
        # Apply edge dropout during training
        if self.training and edge_dropout > 0:
            edge_index = apply_edge_dropout(edge_index, edge_dropout)

        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        # Graph convolution
        embs = [all_emb]
        for conv in self.convs:
            all_emb = conv(all_emb, edge_index)
            embs.append(all_emb)

        # Average pooling across layers
        final_emb = torch.stack(embs, dim=0).mean(dim=0)

        # Split back to user and item
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]

        return user_final, item_final

    def predict_ranking(self, users, items, edge_index, edge_dropout=0.0):
        """
        Predict ranking scores (for BPR loss)

        Returns:
            scores: Ranking scores with bias terms
        """
        user_emb, item_emb = self.forward(edge_index, edge_dropout)
        user_emb = user_emb[users]
        item_emb = item_emb[items]

        # Dot product
        scores = (user_emb * item_emb).sum(dim=1)

        # Add bias terms
        scores = scores + self.user_bias(users).squeeze()
        scores = scores + self.item_bias(items).squeeze()
        scores = scores + self.global_bias

        return scores

    def predict_rating(self, users, items, edge_index, edge_dropout=0.0):
        """
        Predict ratings (for regression loss)

        Returns:
            ratings: Predicted ratings (1~5 scale)
        """
        user_emb, item_emb = self.forward(edge_index, edge_dropout)
        user_emb = user_emb[users]
        item_emb = item_emb[items]

        # Concatenate and pass through MLP
        concat = torch.cat([user_emb, item_emb], dim=1)
        rating = self.rating_mlp(concat)

        # Add bias terms
        rating = rating + self.user_bias(users)
        rating = rating + self.item_bias(items)
        rating = rating + self.global_bias

        return rating.squeeze()

print("\n" + "=" * 80)
print("V8 Model Architecture Defined")
print("=" * 80)
print("✅ User/Item Bias terms added")
print("✅ Multi-task: Rating regression + BPR ranking")
print("✅ Edge dropout for graph augmentation")

# ============================================================================
# V8 Loss Function
# ============================================================================

def v8_multitask_loss(model, edge_index,
                      pos_users, pos_items, pos_ratings,
                      neg_users, neg_items,
                      alpha=0.3, beta=0.7, neg_ratio=1, edge_dropout=0.0):
    """
    V8 Multi-task Loss

    Args:
        alpha: Weight for rating regression loss
        beta: Weight for BPR ranking loss
        neg_ratio: Number of negatives per positive
        edge_dropout: Edge dropout rate

    Returns:
        total_loss, mse_loss, bpr_loss
    """
    # Task 1: Rating Regression (MSE)
    pred_ratings = model.predict_rating(pos_users, pos_items, edge_index, edge_dropout)
    mse_loss = F.mse_loss(pred_ratings, pos_ratings)

    # Task 2: BPR Ranking
    pos_scores = model.predict_ranking(pos_users, pos_items, edge_index, edge_dropout)
    neg_scores = model.predict_ranking(neg_users, neg_items, edge_index, edge_dropout)

    # Reshape for multiple negatives
    if neg_ratio > 1:
        batch_size = pos_scores.size(0)
        neg_scores = neg_scores.view(batch_size, neg_ratio)
        pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores_expanded - neg_scores) + 1e-10).mean()
    else:
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    # Combined loss
    total_loss = alpha * mse_loss + beta * bpr_loss

    return total_loss, mse_loss.item(), bpr_loss.item()

print("\n⭐ V8 Multi-task loss function defined!")
print(f"  - Alpha (MSE weight): {CONFIG['alpha']}")
print(f"  - Beta (BPR weight): {CONFIG['beta']}")

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, edge_index, eval_df, user_items_dict, n_items, k=10, device='cpu'):
    """Evaluate model with Precision@K, Recall@K, NDCG@K"""
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model(edge_index.to(device), edge_dropout=0.0)

        precisions, recalls, ndcgs = [], [], []

        for user_id, group in eval_df.groupby('user_id'):
            true_items = set(group['item_id'].values)
            exclude_items = user_items_dict[user_id]

            # Get scores
            user_emb_single = user_emb[user_id].unsqueeze(0)
            scores = torch.matmul(user_emb_single, item_emb.t()).squeeze()

            # Add bias
            scores = scores + model.item_bias.weight.squeeze()
            scores = scores + model.user_bias.weight[user_id]
            scores = scores + model.global_bias

            # Exclude training items
            scores_np = scores.cpu().numpy()
            for item_id in exclude_items:
                scores_np[int(item_id)] = -np.inf

            # Top-K items
            top_k_items = np.argsort(scores_np)[-k:][::-1]

            # Metrics
            hits = len(set(top_k_items) & true_items)

            precision = hits / k
            recall = hits / len(true_items) if len(true_items) > 0 else 0

            dcg = sum([1 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in true_items])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
            ndcg = dcg / idcg if idcg > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'ndcg@{k}': np.mean(ndcgs),
    }

print("\n✅ Evaluation function defined!")

# ============================================================================
# Training Function
# ============================================================================

def train_one_epoch(model, edge_index, train_df, user_items_dict, n_items,
                    low_rating_items_dict, optimizer, config, device):
    """Train for one epoch"""
    model.train()

    # Shuffle data
    train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

    total_loss = 0
    total_mse = 0
    total_bpr = 0
    n_batches = 0

    for start_idx in range(0, len(train_df_shuffled), config['batch_size']):
        end_idx = min(start_idx + config['batch_size'], len(train_df_shuffled))
        batch_df = train_df_shuffled.iloc[start_idx:end_idx]

        # Positive samples
        pos_users = torch.tensor(batch_df['user_id'].values, dtype=torch.long).to(device)
        pos_items = torch.tensor(batch_df['item_id'].values, dtype=torch.long).to(device)
        pos_ratings = torch.tensor(batch_df['rating'].values, dtype=torch.float).to(device)

        # ⭐ V8: Hard negative sampling
        neg_users_np, neg_items_np = hard_negative_sampling(
            batch_df, user_items_dict, n_items, low_rating_items_dict,
            neg_ratio=config['neg_ratio'], hard_ratio=config['hard_neg_ratio']
        )
        neg_users = torch.tensor(neg_users_np, dtype=torch.long).to(device)
        neg_items = torch.tensor(neg_items_np, dtype=torch.long).to(device)

        # ⭐ V8: Multi-task loss
        loss, mse_loss, bpr_loss = v8_multitask_loss(
            model, edge_index.to(device),
            pos_users, pos_items, pos_ratings,
            neg_users, neg_items,
            alpha=config['alpha'], beta=config['beta'],
            neg_ratio=config['neg_ratio'],
            edge_dropout=config['edge_dropout']
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss
        total_bpr += bpr_loss
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'bpr': total_bpr / n_batches,
    }

print("\n✅ Training function defined!")

# ============================================================================
# Main Training Loop
# ============================================================================

print("\n" + "=" * 80)
print("Training V8 Model")
print("=" * 80)

# Initialize model
model = LightGCN_V8(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=CONFIG['embedding_dim'],
    n_layers=CONFIG['n_layers']
).to(CONFIG['device'])

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: LightGCN V8")
print(f"  - Users: {n_users}, Items: {n_items}")
print(f"  - Embedding dim: {CONFIG['embedding_dim']}")
print(f"  - Layers: {CONFIG['n_layers']}")
print(f"  - Total params: {total_params:,}")
print(f"  - ⭐ With User/Item Bias")
print(f"  - ⭐ With Rating Regression Head")
print(f"\nTraining:")
print(f"  - Device: {CONFIG['device']}")
print(f"  - Batch size: {CONFIG['batch_size']}")
print(f"  - Learning rate: {CONFIG['learning_rate']}")
print(f"  - Weight decay: {CONFIG['weight_decay']}")
print(f"  - Alpha (MSE): {CONFIG['alpha']}")
print(f"  - Beta (BPR): {CONFIG['beta']}")
print(f"  - Negative ratio: {CONFIG['neg_ratio']}")
print(f"  - Hard negative ratio: {CONFIG['hard_neg_ratio']}")
print(f"  - Edge dropout: {CONFIG['edge_dropout']}")
print(f"  - Patience: {CONFIG['patience']}")
print("=" * 80)

# Training history
history = {
    'train_loss': [],
    'train_mse': [],
    'train_bpr': [],
    'valid_precision': [],
    'valid_recall': [],
    'valid_ndcg': [],
}

best_recall = 0
patience_counter = 0

print("\nStarting training...")
print("-" * 80)

for epoch in range(CONFIG['epochs']):
    # Train
    train_metrics = train_one_epoch(
        model, train_edge_index, train_df, train_user_items, n_items,
        low_rating_items_dict, optimizer, CONFIG, CONFIG['device']
    )

    # Validate
    val_metrics = evaluate_model(
        model, train_edge_index, valid_df, train_user_items,
        n_items, k=CONFIG['top_k'], device=CONFIG['device']
    )

    # Record history
    history['train_loss'].append(train_metrics['loss'])
    history['train_mse'].append(train_metrics['mse'])
    history['train_bpr'].append(train_metrics['bpr'])
    history['valid_precision'].append(val_metrics[f'precision@{CONFIG["top_k"]}'])
    history['valid_recall'].append(val_metrics[f'recall@{CONFIG["top_k"]}'])
    history['valid_ndcg'].append(val_metrics[f'ndcg@{CONFIG["top_k"]}'])

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Loss: {train_metrics['loss']:.4f} "
              f"(MSE: {train_metrics['mse']:.4f}, BPR: {train_metrics['bpr']:.4f}) | "
              f"P@{CONFIG['top_k']}: {val_metrics[f'precision@{CONFIG["top_k"]}']:.4f} | "
              f"R@{CONFIG['top_k']}: {val_metrics[f'recall@{CONFIG["top_k"]}']:.4f} | "
              f"NDCG@{CONFIG['top_k']}: {val_metrics[f'ndcg@{CONFIG["top_k"]}']:.4f}")

    # Early stopping
    current_recall = val_metrics[f'recall@{CONFIG["top_k"]}']
    if current_recall > best_recall:
        best_recall = current_recall
        patience_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(CONFIG['model_dir'], 'lightgcn_v8_best.pth'))
    else:
        patience_counter += 1

    if patience_counter >= CONFIG['patience']:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("=" * 80)
print(f"Training Complete!")
print(f"Best Validation Recall@{CONFIG['top_k']}: {best_recall:.4f}")
print("=" * 80)

# ============================================================================
# Test Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("Test Set Evaluation")
print("=" * 80)

# Load best model
model.load_state_dict(torch.load(os.path.join(CONFIG['model_dir'], 'lightgcn_v8_best.pth')))

for k in [5, 10, 20]:
    test_metrics = evaluate_model(
        model,
        train_valid_edge_index,
        test_df,
        train_valid_user_items,
        n_items,
        k=k,
        device=CONFIG['device']
    )

    print(f"\nTop-{k} Recommendations:")
    print(f"  Precision@{k}: {test_metrics[f'precision@{k}']:.4f}")
    print(f"  Recall@{k}:    {test_metrics[f'recall@{k}']:.4f}")
    print(f"  NDCG@{k}:      {test_metrics[f'ndcg@{k}']:.4f}")

print("=" * 80)

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating training curves...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss components
axes[0, 0].plot(history['train_loss'], label='Total Loss', linewidth=2)
axes[0, 0].plot(history['train_mse'], label='MSE Loss', linewidth=2, alpha=0.7)
axes[0, 0].plot(history['train_bpr'], label='BPR Loss', linewidth=2, alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('V8 Training Loss Components')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation metrics
axes[0, 1].plot(history['valid_precision'], label=f'Precision@{CONFIG["top_k"]}', linewidth=2)
axes[0, 1].plot(history['valid_recall'], label=f'Recall@{CONFIG["top_k"]}', linewidth=2)
axes[0, 1].plot(history['valid_ndcg'], label=f'NDCG@{CONFIG["top_k"]}', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('V8 Validation Metrics')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Loss ratio
axes[1, 0].plot([m / (m + b) if (m + b) > 0 else 0
                 for m, b in zip(history['train_mse'], history['train_bpr'])],
                label='MSE / (MSE + BPR)', linewidth=2)
axes[1, 0].axhline(y=CONFIG['alpha'], color='r', linestyle='--',
                   label=f'Target Alpha={CONFIG["alpha"]}')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Ratio')
axes[1, 0].set_title('Loss Component Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall improvement tracking
best_recalls = [max(history['valid_recall'][:i+1]) for i in range(len(history['valid_recall']))]
axes[1, 1].plot(history['valid_recall'], label='Current Recall', linewidth=2, alpha=0.5)
axes[1, 1].plot(best_recalls, label='Best Recall', linewidth=2)
axes[1, 1].axhline(y=0.08, color='r', linestyle='--', label='Baseline (BPR-MF)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel(f'Recall@{CONFIG["top_k"]}')
axes[1, 1].set_title('V8 Recall Improvement')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG['result_dir'], 'training_curves_v8.png'),
            dpi=300, bbox_inches='tight')
print(f"✅ Saved: {os.path.join(CONFIG['result_dir'], 'training_curves_v8.png')}")

print("\n" + "=" * 80)
print("V8 BREAKTHROUGH - Training Complete!")
print("=" * 80)
print(f"Best model saved to: {os.path.join(CONFIG['model_dir'], 'lightgcn_v8_best.pth')}")
print("=" * 80)
