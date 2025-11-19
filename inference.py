"""
CCC22 Inference Script
교수님이 제공하는 test.csv를 받아서 추론하는 스크립트

사용법:
    python inference.py test.csv

출력:
    predictions.csv 파일 생성
    콘솔에 정해진 양식으로 출력
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import json
import sys

# ============================================================================
# 0. Configuration
# ============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# ============================================================================
# 1. Load Training Data (for mappings and user_train_items)
# ============================================================================

print("Loading training data for mappings...")
df = pd.read_csv('data/train.csv')

# Create mappings
user2idx = {u: i for i, u in enumerate(sorted(df['user'].unique()))}
item2idx = {it: i for i, it in enumerate(sorted(df['item'].unique()))}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)

print(f"Users: {n_users}, Items: {n_items}")

# User별 interaction count (for MIN_K)
user_interaction_count = df.groupby('user').size().to_dict()

# User별 train items (for filtering)
df['user_idx'] = df['user'].map(user2idx)
df['item_idx'] = df['item'].map(item2idx)

user_train_items = defaultdict(set)
for u, i in zip(df['user_idx'].values, df['item_idx'].values):
    user_train_items[int(u)].add(int(i))

# User별 K 값
MAX_K = 100

def get_k_for_user(user_id):
    count = user_interaction_count.get(user_id, 0)
    if count <= 10:
        return 2
    k = max(2, int(count * 0.2))
    return min(k, MAX_K)

user_k = {user2idx[u]: get_k_for_user(u) for u in user2idx.keys()}

print(f"✓ Loaded {len(user_train_items)} users' training data")

# ============================================================================
# 2. Load Pretrained Models
# ============================================================================

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index, edge_weight):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs), dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]


class LightGCN_with_Rating(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.rating_mlp = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, edge_index, edge_weight):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs), dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]

    def predict_rating(self, user_idx, item_idx, edge_index, edge_weight):
        u_emb, i_emb = self.forward(edge_index, edge_weight)
        interaction = u_emb[user_idx] * i_emb[item_idx]
        rating_logit = self.rating_mlp(interaction).squeeze(-1)
        predicted_rating = torch.sigmoid(rating_logit) * 4.5 + 0.5
        return predicted_rating


# Build graphs (from train data)
def build_graphs():
    # Train data split from df
    GOOD_RATING_THRESHOLD = 4.0
    train_data = []

    for user_id in df['user'].unique():
        user_idx = user2idx[user_id]
        user_df = df[df['user'] == user_id]

        good_purchases = user_df[user_df['rating'] >= GOOD_RATING_THRESHOLD][['user_idx', 'item_idx']]
        bad_purchases = user_df[user_df['rating'] < GOOD_RATING_THRESHOLD][['user_idx', 'item_idx']]

        if len(bad_purchases) > 0:
            train_data.append(bad_purchases)

        n_good = len(good_purchases)
        if n_good >= 3:
            train_end = int(0.7 * n_good)
            train_data.append(good_purchases.iloc[:train_end])
        elif n_good >= 1:
            train_data.append(good_purchases.iloc[:1])

    train_df = pd.concat(train_data, ignore_index=True)

    # Build unweighted graph (for CCA)
    users = train_df['user_idx'].values
    items = train_df['item_idx'].values

    edge_u2i = np.array([users, items + n_users])
    edge_i2u = np.array([items + n_users, users])
    edge_index = torch.LongTensor(np.concatenate([edge_u2i, edge_i2u], axis=1))

    num_nodes = n_users + n_items
    deg = torch.zeros(num_nodes).scatter_add(0, edge_index[0], torch.ones(edge_index.shape[1]))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    cca_edge_index = edge_index.to(device)
    cca_edge_weight = edge_weight.to(device)

    # Build rating-weighted graph (for CCB)
    ratings = []
    for u, i in zip(users, items):
        user_id = idx2user[u]
        item_id = idx2item[i]
        rating = df[(df['user'] == user_id) & (df['item'] == item_id)]['rating'].values
        ratings.append(rating[0] if len(rating) > 0 else 3)
    ratings = np.array(ratings)

    rating_factors = 0.4 + 0.15 * ratings
    rating_factors_both = np.concatenate([rating_factors, rating_factors])

    base_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    rating_weight = torch.FloatTensor(rating_factors_both)
    edge_weight = base_weight * rating_weight

    ccb_edge_index = edge_index.to(device)
    ccb_edge_weight = edge_weight.to(device)

    return cca_edge_index, cca_edge_weight, ccb_edge_index, ccb_edge_weight


print("Building graphs...")
cca_edge_index, cca_edge_weight, ccb_edge_index, ccb_edge_weight = build_graphs()

# Load models
print("Loading pretrained models...")
EMB_DIM = 32
N_LAYERS = 2

cca_model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
cca_model.load_state_dict(torch.load('cc_models/cca2_best.pt', map_location=device))
cca_model.eval()

ccb_model = LightGCN_with_Rating(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
ccb_model.load_state_dict(torch.load('cc_models/ccb2_best.pt', map_location=device))
ccb_model.eval()

print("✓ Models loaded")

# ============================================================================
# 3. Load Optimal Parameters
# ============================================================================

print("Loading optimal parameters...")
with open('cc_models/ccc21_params.json', 'r') as f:
    params = json.load(f)

ALPHA = params['alpha']
BETA = params['beta']
THRESHOLD = params['threshold']
CCA_MIN = params['cca_min']
CCA_MAX = params['cca_max']
CCB_MIN = params['ccb_min']
CCB_MAX = params['ccb_max']

print(f"  α={ALPHA:.1f}, β={BETA:.1f}, threshold={THRESHOLD:.4f}")

# ============================================================================
# 4. Normalization Functions
# ============================================================================

def normalize_cca_score(score):
    if CCA_MAX == CCA_MIN:
        return 0.5
    normalized = (score - CCA_MIN) / (CCA_MAX - CCA_MIN)
    return np.clip(normalized, 0, 1)

def normalize_ccb_rating(rating):
    return np.clip((rating - CCB_MIN) / (CCB_MAX - CCB_MIN), 0, 1)

# ============================================================================
# 5. Inference Function
# ============================================================================

def predict(test_csv_path):
    """
    Test CSV 파일을 받아서 추천 예측 수행

    출력:
        DataFrame with columns: user, item, recommend
    """
    print(f"\nLoading test data: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Get embeddings
    with torch.no_grad():
        cca_u_emb, cca_i_emb = cca_model(cca_edge_index, cca_edge_weight)

    results = []
    stats = {'total_o': 0, 'total_items': 0, 'min_k_triggered': 0}

    # Process each user
    for user in test_df['user'].unique():
        if user not in user2idx:
            # Unknown user - no recommendations
            user_rows = test_df[test_df['user'] == user]
            for _, row in user_rows.iterrows():
                results.append({
                    'user': row['user'],
                    'item': row['item'],
                    'recommend': 'X'
                })
                stats['total_items'] += 1
            continue

        user_idx = user2idx[user]
        user_rows = test_df[test_df['user'] == user]
        train_items_set = user_train_items[user_idx]

        # Collect items to score
        items_to_score = []
        item_info = []

        for _, row in user_rows.iterrows():
            item = row['item']
            stats['total_items'] += 1

            if item not in item2idx:
                # Unknown item - no recommendation
                results.append({
                    'user': user,
                    'item': item,
                    'recommend': 'X'
                })
                continue

            item_idx = item2idx[item]

            if item_idx in train_items_set:
                # Already in training set - no recommendation
                results.append({
                    'user': user,
                    'item': item,
                    'recommend': 'X'
                })
                continue

            items_to_score.append(item_idx)
            item_info.append(item)

        if len(items_to_score) == 0:
            continue

        # Calculate ensemble scores
        ensemble_scores = []

        for idx, item_idx in enumerate(items_to_score):
            with torch.no_grad():
                cca_score = (cca_u_emb[user_idx] * cca_i_emb[item_idx]).sum().item()
                u_t = torch.tensor([user_idx], dtype=torch.long).to(device)
                i_t = torch.tensor([item_idx], dtype=torch.long).to(device)
                ccb_rating = ccb_model.predict_rating(u_t, i_t, ccb_edge_index, ccb_edge_weight).item()

            cca_norm = normalize_cca_score(cca_score)
            ccb_norm = normalize_ccb_rating(ccb_rating)
            final_score = ALPHA * cca_norm + BETA * ccb_norm

            ensemble_scores.append({
                'index': idx,
                'item': item_info[idx],
                'final_score': final_score
            })

        # ★ Hybrid Selection with MIN_K Fallback
        user_count = user_interaction_count.get(user, 0)
        K = user_k[user_idx]
        MIN_K = 2 if user_count <= 10 else 0

        # Filter by threshold
        above_threshold_indices = [i for i, s in enumerate(ensemble_scores) if s['final_score'] >= THRESHOLD]

        # Apply fallback logic
        if len(above_threshold_indices) < MIN_K:
            # Fallback: 점수 상위 MIN_K개 강제 포함
            sorted_indices = sorted(range(len(ensemble_scores)),
                                  key=lambda i: ensemble_scores[i]['final_score'],
                                  reverse=True)
            selected_indices = set(sorted_indices[:MIN_K])
            stats['min_k_triggered'] += 1
        elif len(above_threshold_indices) > K:
            # Top-K from above threshold
            above_scores = [(i, ensemble_scores[i]['final_score']) for i in above_threshold_indices]
            above_scores_sorted = sorted(above_scores, key=lambda x: x[1], reverse=True)
            selected_indices = set([i for i, _ in above_scores_sorted[:K]])
        else:
            # All above threshold (within K limit)
            selected_indices = set(above_threshold_indices)

        # Generate results
        for score_dict in ensemble_scores:
            if score_dict['index'] in selected_indices:
                recommend = 'O'
                stats['total_o'] += 1
            else:
                recommend = 'X'

            results.append({
                'user': user,
                'item': score_dict['item'],
                'recommend': recommend
            })

    results_df = pd.DataFrame(results)

    # Print stats
    print(f"\n✓ Inference complete!")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Recommended: {stats['total_o']}")
    print(f"  Not recommended: {stats['total_items'] - stats['total_o']}")
    print(f"  O ratio: {100 * stats['total_o'] / stats['total_items']:.1f}%")
    if stats['min_k_triggered'] > 0:
        print(f"  MIN_K fallback triggered: {stats['min_k_triggered']} users")

    return results_df, stats

# ============================================================================
# 6. Output Function
# ============================================================================

def print_results(results_df, stats):
    """
    교수님이 요구하신 정해진 양식으로 출력
    """
    print("\n" + "="*20)
    print(f"{'user':<10} {'item':<10} {'recommend':<10}")
    for _, row in results_df.iterrows():
        print(f"{row['user']:<10} {row['item']:<10} {row['recommend']:<10}")
    print("="*20)
    print(f"Total recommends = {stats['total_o']}/{stats['total_items']}")
    print(f"Not recommend = {stats['total_items'] - stats['total_o']}/{stats['total_items']}")

# ============================================================================
# 7. Main
# ============================================================================

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python inference.py <test.csv>")
        print("\nExample:")
        print("  python inference.py data/test.csv")
        sys.exit(1)

    test_csv_path = sys.argv[1]

    # Run inference
    results_df, stats = predict(test_csv_path)

    # Print in required format
    print_results(results_df, stats)

    # Save to CSV
    output_path = "predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
