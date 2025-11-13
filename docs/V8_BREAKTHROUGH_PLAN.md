# V8 BREAKTHROUGH PLAN

## 현재 상황 분석

### 기존 실험 결과 (V1-V5)
| 모델 | Recall@10 | NDCG@10 | 핵심 아이디어 |
|------|-----------|---------|--------------|
| V1 | 0.0000 | 0.0000 | 기본 LightGCN (data leakage) |
| V2 | N/A | N/A | Rating ≥4.0, 큰 모델 (평가 실패) |
| V3 | 0.0786 | 0.1303 | 작은 모델, rating ≥3.5 |
| V4 | 0.0775 | 0.1331 | neg_ratio 4 |
| V5 | 0.0784 | 0.1340 | 표준 설정 (emb 64, layer 2) |
| **BPR-MF** | **0.0800** | **0.1389** | Graph 없는 MF (최고) |

### 핵심 문제점
1. **절대적 성능 낮음**: Recall@10이 8% 수준 (10개 추천해서 0.8개만 맞춤)
2. **Graph가 도움 안됨**: LightGCN이 BPR-MF보다 낮음
3. **Cold-start 예상**: Long-tail 심각 (79% 아이템이 ≤10 interactions)
4. **Extreme sparsity**: 98.48% → Message passing 제한적

---

## V8 BREAKTHROUGH 전략

### 핵심 아이디어: "Triple Boost Strategy"

#### 🚀 Boost #1: User/Item Bias Modeling
**문제**: 사용자마다 평점 스케일이 다름 (어떤 사람은 후하게, 어떤 사람은 박하게 평가)

**해결책**:
```python
score = user_emb · item_emb + user_bias + item_bias + global_bias
```

**기대 효과**:
- 개인별 평점 경향 보정
- 인기 아이템 편향 명시적 모델링
- Matrix Factorization의 핵심 요소 추가

---

#### 🚀 Boost #2: Multi-Task Learning (Rating Regression + Ranking)
**문제**: Rating 정보를 threshold로만 사용 → 정보 손실

**해결책**:
```python
# Task 1: Rating Regression
rating_pred = model.predict_rating(user, item)
mse_loss = (rating_pred - true_rating)^2

# Task 2: BPR Ranking
bpr_loss = -log(sigmoid(pos_score - neg_score))

# Combined
total_loss = alpha * mse_loss + (1 - alpha) * bpr_loss
```

**기대 효과**:
- Rating의 연속적 정보 활용
- 더 풍부한 학습 신호
- Regression이 representation 학습 도움

---

#### 🚀 Boost #3: Hard Negative Sampling
**문제**: Random negative는 너무 쉬움 → 학습 신호 약함

**해결책**:
```python
# Strategy 1: Low-rating items as negative (rating < 3.0)
# Strategy 2: Popular but not interacted (hard to distinguish)
# Strategy 3: Mixed (50% hard, 50% random)
```

**기대 효과**:
- 더 어려운 negative로 더 강한 학습
- 실제 dislike 정보 활용
- Better discrimination

---

### 추가 개선사항

#### 4. Graph Augmentation
```python
# Training time에 edge dropout 적용
edge_dropout_rate = 0.1  # 10% edge를 랜덤하게 제거
# Robustness 향상, overfitting 방지
```

#### 5. Attention Mechanism (선택적)
```python
# 중요한 neighbor에 더 집중
# GraphSAGE-style attention 또는 GAT
```

#### 6. Layer Normalization
```python
# 각 layer 후 normalization
# Training stability 향상
```

---

## V8 Model Architecture

```python
class LightGCN_V8(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # ⭐ Bias terms (Boost #1)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Graph layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])

        # ⭐ Rating regression head (Boost #2)
        self.rating_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, edge_index, edge_dropout=0.0):
        # Edge dropout for augmentation
        if self.training and edge_dropout > 0:
            edge_index = apply_edge_dropout(edge_index, edge_dropout)

        # Graph convolution
        user_emb, item_emb = self.graph_forward(edge_index)

        return user_emb, item_emb

    def predict_ranking(self, users, items, edge_index):
        """For BPR ranking loss"""
        user_emb, item_emb = self.forward(edge_index)
        user_emb = user_emb[users]
        item_emb = item_emb[items]

        # Dot product + bias
        scores = (user_emb * item_emb).sum(dim=1)
        scores = scores + self.user_bias(users).squeeze()
        scores = scores + self.item_bias(items).squeeze()
        scores = scores + self.global_bias

        return scores

    def predict_rating(self, users, items, edge_index):
        """For rating regression"""
        user_emb, item_emb = self.forward(edge_index)
        user_emb = user_emb[users]
        item_emb = item_emb[items]

        # MLP for rating prediction
        concat = torch.cat([user_emb, item_emb], dim=1)
        rating = self.rating_mlp(concat)

        # Add bias
        rating = rating + self.user_bias(users)
        rating = rating + self.item_bias(items)
        rating = rating + self.global_bias

        return rating.squeeze()
```

---

## V8 Training Strategy

### Loss Function
```python
def v8_loss(model, pos_users, pos_items, pos_ratings,
            neg_users, neg_items, edge_index, alpha=0.3):
    """
    alpha: weight for rating regression (0.3)
    1-alpha: weight for BPR ranking (0.7)
    """

    # Boost #2: Rating Regression
    pred_ratings = model.predict_rating(pos_users, pos_items, edge_index)
    mse_loss = F.mse_loss(pred_ratings, pos_ratings)

    # BPR Ranking
    pos_scores = model.predict_ranking(pos_users, pos_items, edge_index)
    neg_scores = model.predict_ranking(neg_users, neg_items, edge_index)
    bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    # Combined
    total_loss = alpha * mse_loss + (1 - alpha) * bpr_loss

    return total_loss, mse_loss.item(), bpr_loss.item()
```

### Hard Negative Sampling (Boost #3)
```python
def hard_negative_sampling(batch_df, user_items_dict, n_items,
                           low_rating_items_dict, hard_ratio=0.5):
    """
    hard_ratio: 50%는 hard negative, 50%는 random negative
    """
    neg_users, neg_items = [], []

    for user_id, pos_item in zip(batch_df['user_id'], batch_df['item_id']):
        user_pos_items = user_items_dict[user_id]
        user_low_rating = low_rating_items_dict.get(user_id, set())

        # Hard negative (from low ratings)
        if random.random() < hard_ratio and len(user_low_rating) > 0:
            neg_item = random.choice(list(user_low_rating))
        # Random negative
        else:
            while True:
                neg_item = random.randint(0, n_items - 1)
                if neg_item not in user_pos_items:
                    break

        neg_users.append(user_id)
        neg_items.append(neg_item)

    return np.array(neg_users), np.array(neg_items)
```

### Hyperparameters (V8)
```python
CONFIG = {
    # Model
    'embedding_dim': 64,
    'n_layers': 2,

    # Training
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 100,
    'patience': 20,  # 더 여유있게

    # Loss weights (⭐ NEW)
    'alpha': 0.3,  # Rating regression weight

    # Negative sampling (⭐ NEW)
    'neg_ratio': 4,
    'hard_neg_ratio': 0.5,  # 50% hard negative

    # Augmentation (⭐ NEW)
    'edge_dropout': 0.1,  # 10% edge dropout

    # Rating threshold for low ratings
    'low_rating_threshold': 3.0,  # < 3.0은 hard negative
    'high_rating_threshold': 3.5,  # ≥ 3.5는 positive
}
```

---

## Expected Performance Gain

### Baseline
- BPR-MF: Recall@10 = 0.0800, NDCG@10 = 0.1389
- LightGCN V5: Recall@10 = 0.0784, NDCG@10 = 0.1340

### Target (V8)
- **Conservative**: Recall@10 > 0.09 (+12.5% improvement)
- **Optimistic**: Recall@10 > 0.10 (+25% improvement)
- **Ambitious**: Recall@10 > 0.12 (+50% improvement)

### 개선 요인별 기대 효과
1. User/Item Bias: +1~2% (개인별 선호도 보정)
2. Multi-task Learning: +1~2% (rating 정보 활용)
3. Hard Negative Sampling: +1~2% (더 강한 학습 신호)
4. Graph Augmentation: +0.5~1% (robustness)
5. **Combined Synergy**: +2~3% (시너지 효과)

**Total Expected**: +5~10% absolute improvement

---

## Implementation Plan

### Phase 1: Core Implementation
1. ✅ Breakthrough 분석
2. ⬜ Low-rating 데이터 준비 (rating < 3.0)
3. ⬜ LightGCN_V8 모델 구현
4. ⬜ Multi-task loss 구현
5. ⬜ Hard negative sampling 구현

### Phase 2: Training
6. ⬜ Training loop 구현
7. ⬜ Validation 평가
8. ⬜ Hyperparameter tuning (alpha, hard_neg_ratio, etc.)

### Phase 3: Evaluation & Analysis
9. ⬜ Test set 평가
10. ⬜ Ablation study (각 요소별 기여도)
11. ⬜ 결과 시각화

### Phase 4: Further Improvements (시간 있으면)
12. ⬜ Attention mechanism 추가
13. ⬜ Ensemble (V8 + BPR-MF)
14. ⬜ 다양한 neg_ratio, alpha 값 실험

---

## Success Criteria

### Minimum Success
- [x] V8 구현 완료
- [ ] Recall@10 > 0.08 (baseline 수준)
- [ ] NDCG@10 > 0.13 (baseline 수준)

### Expected Success
- [ ] Recall@10 > 0.09 (+12.5%)
- [ ] NDCG@10 > 0.14 (+7%)

### Outstanding Success
- [ ] Recall@10 > 0.10 (+25%)
- [ ] NDCG@10 > 0.15 (+15%)
- [ ] Ablation study로 각 요소 기여도 확인

---

## Fallback Plan

만약 V8이 큰 개선을 보이지 못하면:

### Plan B: Ensemble Approach
```python
# Simple weighted average
final_score = 0.5 * lightgcn_score + 0.5 * bpr_mf_score

# Learned ensemble
ensemble_score = mlp([lightgcn_score, bpr_mf_score, popularity_score])
```

### Plan C: Re-ranking
```python
# Diversity-aware re-ranking
# Coverage-aware re-ranking
# Calibration
```

### Plan D: Different Architecture
- GraphSAGE with LSTM aggregator
- GAT with multi-head attention
- NGCF (Neural Graph Collaborative Filtering)

---

## Timeline

- **Phase 1 (2h)**: Implementation
- **Phase 2 (2h)**: Training & tuning
- **Phase 3 (1h)**: Evaluation
- **Total**: 5 hours

---

## Notes

- BPR-MF가 LightGCN보다 좋다는 것은 graph structure가 이 데이터셋에서 충분히 활용되지 못했다는 의미
- User/Item bias는 MF의 핵심 요소이므로 추가하면 LightGCN도 MF 수준 성능 기대
- Multi-task learning은 rating의 rich information을 활용하므로 효과적일 것
- Hard negative는 특히 sparse data에서 효과적

---

**Created**: 2025-11-13
**Status**: Ready to implement
**Expected Completion**: 2025-11-13 EOD
