# Non-GNN Deep Learning 추천 시스템 전략

## 개요

GNN을 사용하지 않고 **전통적인 Deep Learning** 방식으로 추천 시스템을 구축하는 방법을 탐구한다. 그래프 구조를 명시적으로 활용하지 않고도 강력한 성능을 낼 수 있는 접근법들이 있다.

---

## 1. 왜 Non-GNN 방식을 고려하는가?

### GNN의 한계
1. **그래프 구조 필수**: 명시적인 edge 정보 필요
2. **계산 복잡도**: Neighbor aggregation이 비용 높음
3. **스케일링 어려움**: 대규모 그래프에선 메모리 문제
4. **Over-smoothing**: 레이어 깊어지면 모든 노드가 비슷해짐

### Non-GNN의 장점
1. **단순성**: 구현 및 디버깅 쉬움
2. **속도**: 추론 시 그래프 연산 불필요
3. **유연성**: 다양한 feature 쉽게 추가
4. **검증된 방법**: 산업계에서 널리 사용됨 (Netflix, YouTube 등)

---

## 2. 주요 Non-GNN 접근법

### 2.1 Neural Collaborative Filtering (NCF) ⭐ 추천

**개념:**
- User와 Item을 각각 embedding으로 표현
- MLP로 interaction 학습
- Matrix Factorization의 neural network 확장

**아키텍처:**
```
Input: (user_id, item_id)
         ↓
    [Embedding Layer]
         ↓
    user_emb [64]  item_emb [64]
         ↓             ↓
    [Concatenate or Element-wise Product]
         ↓
    combined [128] or [64]
         ↓
    [MLP Layers]
    FC1: 128 → 64 (ReLU, Dropout)
    FC2: 64 → 32 (ReLU, Dropout)
    FC3: 32 → 16 (ReLU)
         ↓
    [Output Layer]
    FC4: 16 → 1 (Sigmoid or Linear)
         ↓
    prediction (rating or probability)
```

**두 가지 변형:**

**GMF (Generalized Matrix Factorization):**
```python
# Element-wise product (내적과 유사)
interaction = user_emb * item_emb  # [64]
output = sigmoid(W @ interaction + b)
```

**MLP (Multi-Layer Perceptron):**
```python
# Concatenation + Deep layers
combined = concat([user_emb, item_emb])  # [128]
h1 = ReLU(FC1(combined))
h2 = ReLU(FC2(h1))
output = sigmoid(FC_out(h2))
```

**NeuMF (Neural Matrix Factorization):** GMF + MLP 결합
```python
# 두 경로를 parallel로 학습 후 결합
gmf_out = gmf_layer(user_emb, item_emb)
mlp_out = mlp_layers(concat([user_emb, item_emb]))
final = sigmoid(concat([gmf_out, mlp_out]))
```

**장점:**
- 단순하고 효과적
- Non-linear interaction 학습 가능
- 빠른 학습 및 추론

**단점:**
- User-item pair만 보고 주변 정보 활용 못함
- Cold-start 문제 해결 어려움

**구현 예시:**
```python
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_emb(user_ids)  # [batch, 64]
        item_emb = self.item_emb(item_ids)  # [batch, 64]
        
        # Concatenate
        x = torch.cat([user_emb, item_emb], dim=1)  # [batch, 128]
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Output
        out = self.fc_out(x)  # [batch, 1]
        return out.squeeze()
```

---

### 2.2 AutoEncoder 기반

**개념:**
- User-item interaction matrix를 입력으로 받아 복원
- Latent space에서 user preference 학습
- Denoising을 통해 robustness 향상

**아키텍처:**
```
Input: user vector [10321] (모든 item에 대한 rating, 대부분 0)
         ↓
    [Encoder]
    FC1: 10321 → 512 (ReLU, Dropout)
    FC2: 512 → 128 (ReLU, Dropout)
    FC3: 128 → 64 (Latent representation)
         ↓
    [Decoder]
    FC4: 64 → 128 (ReLU, Dropout)
    FC5: 128 → 512 (ReLU, Dropout)
    FC6: 512 → 10321 (Sigmoid or Linear)
         ↓
    Output: reconstructed ratings [10321]
```

**Variational AutoEncoder (VAE) 변형:**
```python
# Latent space를 확률 분포로 모델링
z_mean = encoder_mean(x)
z_log_var = encoder_logvar(x)
z = sampling(z_mean, z_log_var)  # Reparameterization trick
reconstructed = decoder(z)

# Loss = Reconstruction Loss + KL Divergence
loss = MSE(x, reconstructed) + KL_divergence(z_mean, z_log_var)
```

**장점:**
- 전체 user profile을 한 번에 고려
- Missing data imputation 자연스럽게 처리
- Unsupervised pre-training 가능

**단점:**
- 입력 차원이 매우 큼 (10,321개 item)
- 대부분이 0인 sparse input → 학습 어려움
- 추론 시 전체 아이템 점수 계산 필요

**개선 전략:**
- **Denoising**: Input에 noise 추가해 robustness 향상
- **Weighted loss**: 관측된 값에만 높은 가중치
- **Collaborative Denoising Auto-Encoder (CDAE)**: User ID를 추가 입력으로

---

### 2.3 Deep & Cross Network (DCN)

**개념:**
- Feature interaction을 명시적으로 학습
- Cross network로 high-order interaction 자동 생성
- CTR 예측에서 효과적

**아키텍처:**
```
Input: [user_emb (64), item_emb (64)] → concat [128]
         ↓
    [Cross Network]              [Deep Network]
    x₀ = input                   x = input
    x₁ = x₀ x₀ᵀw₁ + b₁ + x₀      ↓
    x₂ = x₀ x₁ᵀw₂ + b₂ + x₁      FC1: 128 → 64
    ...                          FC2: 64 → 32
    xₗ (cross features)          FC3: 32 → 16
         ↓                            ↓
    [Concatenate: xₗ + deep output]
         ↓
    [Output Layer]
    FC: (128+16) → 1
```

**Cross Layer 수식:**
```
x_{l+1} = x_0 · (x_l^T · w_l) + b_l + x_l

효과: Feature crosses (x_i * x_j, x_i * x_j * x_k 등)를 자동 생성
```

**장점:**
- High-order feature interaction 명시적 학습
- Parameter efficiency (cross layer는 가벼움)

**단점:**
- Cross network 자체는 linear combination
- 복잡도 대비 성능 향상 미미할 수 있음

---

### 2.4 Wide & Deep

**개념:**
- Wide component: Linear model (memorization)
- Deep component: DNN (generalization)
- 두 가지를 결합해 장점 취합

**아키텍처:**
```
        Input Features
             ↓
    ┌────────┴────────┐
    ↓                 ↓
[Wide]            [Deep]
Linear         Embedding + MLP
Model               ↓
    ↓           FC1 → FC2 → FC3
    └────────┬────────┘
             ↓
        [Combined]
        Sigmoid(W_wide + W_deep)
```

**Wide part:**
```python
# User-item pair + cross features
wide_input = [user_id_one_hot, item_id_one_hot, user*item]
wide_out = Linear(wide_input)
```

**Deep part:**
```python
# Embedding + MLP
user_emb = Embedding(user_id)
item_emb = Embedding(item_id)
deep_out = MLP(concat([user_emb, item_emb]))
```

**장점:**
- Memorization (특정 패턴 암기) + Generalization (일반화) 균형
- 산업계에서 검증됨 (Google Play 추천)

**단점:**
- Feature engineering 필요 (cross features)
- 복잡한 아키텍처

---

### 2.5 Simple MLP (Baseline)

**개념:**
- 가장 단순한 접근: User/Item ID → Embedding → MLP

**아키텍처:**
```
Input: (user_id, item_id)
         ↓
    [Embedding]
    user_emb [64], item_emb [64]
         ↓
    [Concatenate] → [128]
         ↓
    FC1: 128 → 64 (ReLU)
    FC2: 64 → 32 (ReLU)
    FC3: 32 → 1 (Linear)
         ↓
    Output: rating prediction
```

**장점:**
- 매우 단순, 빠른 구현
- 디버깅 쉬움

**단점:**
- 표현력 제한적
- 성능이 다른 방법보다 낮을 가능성

---

## 3. Non-GNN vs GNN 비교

| 측면 | Non-GNN (NCF) | GNN (LightGCN) |
|------|---------------|----------------|
| **입력** | (user, item) pair | User-item bipartite graph |
| **정보 활용** | Direct interaction만 | Multi-hop neighborhood |
| **계산 복잡도** | O(batch_size) | O(edges * layers) |
| **추론 속도** | 매우 빠름 | 상대적으로 느림 |
| **Cold-start** | 어려움 | 약간 나음 (neighbor info) |
| **해석가능성** | 낮음 (블랙박스) | 높음 (graph structure) |
| **성능 (일반적)** | 좋음 | 더 좋음 (약 5~10% 향상) |

**핵심 차이:**
- **NCF**: "이 사용자와 이 영화의 조합이 좋은가?" (pair-wise)
- **GNN**: "비슷한 취향의 사용자들이 본 영화를 추천" (graph-wise)

---

## 4. 현재 프로젝트에 Non-GNN 적용 전략

### 4.1 추천 모델: NCF (NeuMF)

**이유:**
1. 단순하지만 강력
2. 빠른 학습 및 추론
3. Negative sampling과 자연스럽게 결합
4. 산업계에서 검증됨

### 4.2 아키텍처 설계

```python
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, 
                 embedding_dim=64, hidden_layers=[128, 64, 32]):
        super().__init__()
        
        # GMF part
        self.user_emb_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP part
        self.user_emb_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.fc_out = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
    def forward(self, user_ids, item_ids):
        # GMF path
        user_gmf = self.user_emb_gmf(user_ids)
        item_gmf = self.item_emb_gmf(item_ids)
        gmf_out = user_gmf * item_gmf  # Element-wise product
        
        # MLP path
        user_mlp = self.user_emb_mlp(user_ids)
        item_mlp = self.item_emb_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_out = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        prediction = self.fc_out(combined)
        return prediction.squeeze()
```

### 4.3 손실 함수

**Option 1: BPR Loss (추천)**
```python
def bpr_loss(pos_scores, neg_scores):
    diff = pos_scores.unsqueeze(1) - neg_scores  # Broadcasting
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss

# Usage
pos_pred = model(users, pos_items)
neg_pred = model(users.unsqueeze(1).expand(-1, K), neg_items)
loss = bpr_loss(pos_pred, neg_pred)
```

**Option 2: Binary Cross-Entropy**
```python
# Positive samples: label=1
# Negative samples: label=0
criterion = nn.BCEWithLogitsLoss()

pos_pred = model(users, pos_items)
neg_pred = model(users, neg_items)

pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
loss = pos_loss + neg_loss
```

### 4.4 학습 파이프라인

```python
# 1. 데이터 준비
train_data = {
    'users': [...],
    'pos_items': [...],
    'neg_items': [...]  # K개 per user
}

# 2. Model 초기화
model = NeuMF(
    num_users=668,
    num_items=10321,
    embedding_dim=64,
    hidden_layers=[128, 64, 32]
)

# 3. Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

# 4. Training loop
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        users, pos_items, neg_items = batch
        
        # Forward
        pos_scores = model(users, pos_items)
        neg_scores = model(users, neg_items)
        
        # Loss
        loss = bpr_loss(pos_scores, neg_scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    if epoch % 5 == 0:
        metrics = evaluate(model, val_data)
        print(f"Epoch {epoch}: Recall@10={metrics['recall']:.4f}")
```

---

## 5. 성능 예측 및 비교

### 5.1 예상 성능

**MovieLens 10M 벤치마크 기준:**

| 모델 | Recall@10 | NDCG@10 | 학습 시간 |
|------|-----------|---------|----------|
| Random | 0.01 | 0.01 | - |
| Popularity | 0.08 | 0.06 | - |
| MF (SVD) | 0.18 | 0.22 | 빠름 |
| NCF (MLP) | 0.22 | 0.26 | 중간 |
| NeuMF | 0.25 | 0.29 | 중간 |
| LightGCN | 0.30 | 0.35 | 느림 |
| LightGCN + Ensemble | 0.35 | 0.40 | 매우 느림 |

**현재 데이터셋 예상:**
- 데이터가 더 작고 희소 → 성능 전반적으로 낮을 것
- **NeuMF 예상: Recall@10 ≈ 0.20~0.25**
- **LightGCN 예상: Recall@10 ≈ 0.25~0.30**
- **Gap: 약 5~10%** (GNN이 근소하게 우세)

### 5.2 장단점 비교

**NeuMF 장점:**
- ✅ 구현 단순 (100줄 이내)
- ✅ 학습 속도 빠름 (2~3배)
- ✅ 추론 속도 매우 빠름 (5~10배)
- ✅ 디버깅 쉬움
- ✅ Production 배포 용이

**NeuMF 단점:**
- ❌ 성능이 GNN보다 5~10% 낮음
- ❌ Cold-start 취약
- ❌ Neighborhood information 활용 못함
- ❌ 해석가능성 낮음

**결론:**
- **시간 제약 있으면**: NeuMF (충분히 좋은 성능)
- **최고 성능 원한다면**: LightGCN
- **Production 고려**: NeuMF (속도 중요)

---

## 6. Hybrid 접근: GNN + Non-GNN

### 6.1 Ensemble 전략

**방법 1: Score Averaging**
```python
score_final = α * score_ncf + (1-α) * score_gnn

# α 찾기: Validation set에서 grid search
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
best_alpha = argmax(recall@10(alpha) for alpha in alphas)
```

**방법 2: Stacking**
```python
# NCF와 GNN의 출력을 feature로 사용
meta_features = [score_ncf, score_gnn, user_avg_rating, item_popularity]
meta_model = LogisticRegression()
meta_model.fit(meta_features, labels)
```

**효과:** 각 모델의 장점 결합 (보통 2~5% 추가 향상)

### 6.2 Two-Stage 전략

**Stage 1: NeuMF로 후보 생성**
```python
# 빠른 NeuMF로 Top-100 후보 추출
candidates = ncf.predict_top_k(user, k=100)
```

**Stage 2: LightGCN으로 re-ranking**
```python
# 정교한 GNN으로 최종 Top-10 선정
final_recommendations = gnn.rerank(user, candidates, k=10)
```

**효과:** 
- 속도와 성능 균형
- Production에서 자주 사용되는 패턴

---

## 7. 실전 가이드: Non-GNN 구현

### 7.1 구현 우선순위

**Week 1:**
1. Simple MLP baseline (1일)
2. NCF (MLP only) (1일)
3. NeuMF (GMF + MLP) (2일)
4. 하이퍼파라미터 튜닝 (3일)

**Week 2:**
1. AutoEncoder 시도 (선택, 2일)
2. Wide & Deep 시도 (선택, 2일)
3. Ensemble (NCF + MF) (1일)
4. 최종 평가 (2일)

### 7.2 디버깅 체크리스트

**학습 중 확인사항:**
- [ ] Training loss 감소하는가?
- [ ] Validation loss가 발산하지 않는가? (overfitting)
- [ ] Gradient exploding/vanishing 없는가?
- [ ] Negative samples가 제대로 생성되는가?
- [ ] Embedding이 학습되는가? (norm 변화 확인)

**성능 확인:**
- [ ] Random baseline보다 높은가?
- [ ] Popularity baseline보다 높은가?
- [ ] Matrix Factorization과 비교는?
- [ ] 모든 사용자에게 같은 추천 하지 않는가?

### 7.3 성능 개선 Tip

**Embedding 초기화:**
```python
# Xavier initialization (추천)
nn.init.xavier_uniform_(model.user_emb.weight)
nn.init.xavier_uniform_(model.item_emb.weight)

# 또는 Normal distribution
nn.init.normal_(model.user_emb.weight, std=0.01)
```

**Dropout 전략:**
```python
# Embedding에는 약하게, MLP에는 강하게
self.emb_dropout = nn.Dropout(0.1)
self.mlp_dropout = nn.Dropout(0.3)
```

**Batch Normalization:**
```python
# MLP layer 사이에 추가 (선택)
self.bn1 = nn.BatchNorm1d(128)
x = self.bn1(F.relu(self.fc1(x)))
```

---

## 8. 최종 권장사항

### 8.1 프로젝트 목표에 따른 선택

**목표: 최고 성능**
→ **LightGCN** (GNN)
- 예상 Recall@10: 0.25~0.30
- 구현 난이도: 중상
- 학습 시간: 길음

**목표: 빠른 프로토타입**
→ **NeuMF** (Non-GNN)
- 예상 Recall@10: 0.20~0.25
- 구현 난이도: 중
- 학습 시간: 짧음

**목표: 균형 (추천)**
→ **NeuMF + LightGCN Ensemble**
- 예상 Recall@10: 0.28~0.33
- 구현 난이도: 중상
- 학습 시간: 중간

### 8.2 Time-Performance Trade-off

```
구현 시간 | 성능 | 추천 모델
---------|------|----------
1주      | 중   | NCF (MLP)
2주      | 중상 | NeuMF
3주      | 상   | LightGCN
4주      | 최상 | LightGCN + Ensemble
```

### 8.3 체크리스트

**Non-GNN 구현 시 필수:**
- [x] User/Item embedding 잘 초기화했는가?
- [x] Negative sampling 제대로 구현했는가?
- [x] BPR Loss 또는 BCE Loss 선택했는가?
- [x] Overfitting 방지 (dropout, weight decay)
- [x] Validation으로 hyperparameter 튜닝
- [x] Baseline (MF)과 성능 비교
- [x] Recall@K, NDCG@K 제대로 구현

---

## 9. 코드 템플릿

### 완전한 NeuMF 예제

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RecDataset(Dataset):
    def __init__(self, users, pos_items, neg_items):
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'pos_item': self.pos_items[idx],
            'neg_items': self.neg_items[idx]  # List of K negatives
        }

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # GMF embeddings
        self.user_emb_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        # Output layer
        self.fc_out = nn.Linear(embedding_dim + 32, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb_gmf.weight)
        nn.init.xavier_uniform_(self.item_emb_gmf.weight)
        nn.init.xavier_uniform_(self.user_emb_mlp.weight)
        nn.init.xavier_uniform_(self.item_emb_mlp.weight)
    
    def forward(self, user_ids, item_ids):
        # GMF part
        user_gmf = self.user_emb_gmf(user_ids)
        item_gmf = self.item_emb_gmf(item_ids)
        gmf_out = user_gmf * item_gmf
        
        # MLP part
        user_mlp = self.user_emb_mlp(user_ids)
        item_mlp = self.item_emb_mlp(item_ids)
        
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_out = F.relu(self.fc1(mlp_input))
        mlp_out = self.dropout(mlp_out)
        mlp_out = F.relu(self.fc2(mlp_out))
        mlp_out = self.dropout(mlp_out)
        mlp_out = F.relu(self.fc3(mlp_out))
        
        # Concatenate and output
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.fc_out(combined)
        return output.squeeze(-1)

def bpr_loss(pos_scores, neg_scores):
    """
    pos_scores: [batch_size]
    neg_scores: [batch_size, num_negatives]
    """
    # Expand pos_scores for broadcasting
    pos_scores = pos_scores.unsqueeze(1)  # [batch_size, 1]
    
    # Compute difference
    diff = pos_scores - neg_scores  # [batch_size, num_negatives]
    
    # BPR loss
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        users = batch['user'].to(device)
        pos_items = batch['pos_item'].to(device)
        neg_items = batch['neg_items'].to(device)  # [batch, K]
        
        # Positive scores
        pos_scores = model(users, pos_items)
        
        # Negative scores
        batch_size, K = neg_items.shape
        users_expanded = users.unsqueeze(1).expand(-1, K)
        neg_scores = model(users_expanded, neg_items)
        
        # Loss
        loss = bpr_loss(pos_scores, neg_scores)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, users, items, labels, k=10, device='cpu'):
    """
    Compute Recall@K
    """
    model.eval()
    recalls = []
    
    with torch.no_grad():
        for user in users:
            # Get all item scores for this user
            user_tensor = torch.tensor([user] * len(items)).to(device)
            items_tensor = torch.tensor(items).to(device)
            
            scores = model(user_tensor, items_tensor)
            
            # Get top-K
            _, top_k_indices = torch.topk(scores, k)
            top_k_items = [items[i] for i in top_k_indices.cpu().numpy()]
            
            # Compute recall
            relevant = [item for item, label in zip(items, labels) if label == 1]
            hit = len(set(top_k_items) & set(relevant))
            recall = hit / len(relevant) if len(relevant) > 0 else 0
            recalls.append(recall)
    
    return np.mean(recalls)

# Main training loop
if __name__ == '__main__':
    # Hyperparameters
    num_users = 668
    num_items = 10321
    embedding_dim = 64
    batch_size = 256
    epochs = 100
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = NeuMF(num_users, num_items, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        if epoch % 5 == 0:
            val_recall = evaluate(model, val_users, val_items, val_labels, k=10, device=device)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Recall@10={val_recall:.4f}")
```

---

## 10. 결론

### Non-GNN은 언제 선택하는가?

**선택해야 할 때:**
1. 구현 시간이 부족할 때 (1~2주)
2. 추론 속도가 중요할 때 (Production)
3. 간단한 baseline이 필요할 때
4. GNN 라이브러리 설치가 어려울 때
5. 디버깅이 중요할 때

**선택하지 말아야 할 때:**
1. 최고 성능이 필수일 때
2. 그래프 구조가 명확할 때
3. Cold-start 문제가 심각할 때
4. 학습 시간이 충분할 때

### 최종 요약

**Non-GNN (NeuMF):**
- 장점: 빠르고, 간단하고, 안정적
- 단점: 성능이 GNN보다 5~10% 낮음
- 추천 대상: 시간 제약이 있거나 Production 고려

**GNN (LightGCN):**
- 장점: 최고 성능, 그래프 정보 활용
- 단점: 복잡하고, 느리고, 디버깅 어려움
- 추천 대상: 최고 성능이 목표이고 시간 충분

**Hybrid (NeuMF + LightGCN):**
- 장점: 최상의 성능, 각 모델의 장점 결합
- 단점: 구현 및 학습 시간 2배
- 추천 대상: 시간 충분하고 최고 점수 원할 때

**프로젝트 추천 전략:**
1. 먼저 NeuMF 빠르게 구현 (1주)
2. Baseline 성능 확인
3. 만족스럽다면 → 최적화 및 ensemble
4. 부족하다면 → LightGCN 추가 구현