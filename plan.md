# GNN RecSys 개선 계획 (V9 이후)

## 현재 상태 (2025-11-18)

### V9a 성능
- **AUC-ROC**: 0.8913 (+1.18% vs V8)
- **F1 Score**: 0.8398 (+0.56% vs V8)
- **Precision**: 0.8710
- **Recall**: 0.8107
- **Parameters**: 351,648 (50% 감소)
- **Ground Truth**: 모든 구매

### V9b 성능 ⭐ BEST
- **AUC-ROC**: 0.9263 (+0.69% vs V8b)
- **F1 Score**: 0.8719 (+0.68% vs V8b)
- **Precision**: 0.8646
- **Recall**: 0.8794
- **Accuracy**: 0.8708
- **O Ratio**: 50.9%
- **Ground Truth**: Rating >= 4인 구매만

### 적용된 Breakthrough
1. ✅ **Hard Negative Mining** (50% hard, 50% random)
   - Informative한 negative 샘플 학습
   - Loss 증가하지만 일반화 성능 향상

2. ✅ **Rating Weighted Graph**
   - Rating factor: 0.4 + 0.15 * rating
   - 높은 rating에 더 큰 edge weight
   - Signal quality 향상

3. ✅ **Small Embedding** (EMB_DIM=32)
   - 파라미터 50% 감소 (703K → 351K)
   - Overfitting 방지
   - 학습 시간 단축 (5.5분 → 3.6분)

---

## 성능 진화 히스토리

```
버전    AUC-ROC    F1      특징
V4b     0.8968    0.8414  F1 Maximization Threshold
V8      0.8795    0.8342  Hybrid (Threshold + Top-K)
V8b     0.9194    0.8651  Rating-aware (Rating >= 4)
V9a     0.8913    0.8398  + Hard Negative + Rating Weighted
V9b     0.9263    0.8719  + Hard Negative + Rating Weighted (BEST)
```

**누적 개선 (V4b → V9b):**
- AUC-ROC: +2.95%p
- F1: +3.05%p

---

## V10 계획: Item-Item Co-occurrence Graph

### 목표
- **Target AUC-ROC**: 0.93+ (현재 0.9263)
- **Target F1**: 0.88+ (현재 0.8719)
- **예상 향상**: +3-7% Recall

### 핵심 아이디어

**현재 문제:**
- User-Item bipartite graph만 사용
- Item 간 유사성 정보 미활용

**해결책:**
Item-Item similarity graph 추가 구축
```
같이 구매되는 item끼리 연결
→ "이 item을 산 사람은 저 item도 샀다"
→ Collaborative filtering의 "item-based" 관점 추가
```

### 구현 방법

#### 1. Item-Item Graph 구축
```python
def build_item_item_graph(train_df, n_items, k_neighbors=10):
    """
    Co-purchase 패턴 기반 item-item 유사도
    """
    # Item-User matrix
    item_user_matrix = sparse.csr_matrix(
        (np.ones(len(train_df)),
         (train_df['item_idx'], train_df['user_idx'])),
        shape=(n_items, n_users)
    )

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    item_sim = cosine_similarity(item_user_matrix)

    # k-NN graph (각 item의 top-k similar items)
    edges, weights = [], []
    for i in range(n_items):
        neighbors = np.argsort(item_sim[i])[-k_neighbors-1:-1]
        for j in neighbors:
            if i != j:
                edges.append([i, j])
                weights.append(item_sim[i, j])

    return torch.LongTensor(edges).T, torch.FloatTensor(weights)
```

#### 2. Dual-Graph LightGCN 모델
```python
class DualGraphLightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # UI/II graph mixing

    def forward(self, ui_edge_index, ui_weight, ii_edge_index, ii_weight):
        # User-Item propagation
        u_emb, i_emb = self.ui_propagation(ui_edge_index, ui_weight)

        # Item-Item propagation (refine item embeddings)
        i_emb_refined = self.ii_propagation(i_emb, ii_edge_index, ii_weight)

        # Adaptive mixing
        i_emb_final = self.alpha * i_emb + (1 - self.alpha) * i_emb_refined

        return u_emb, i_emb_final
```

#### 3. 학습 파이프라인
```python
# Graph 구축
ui_edge_index, ui_weight = build_rating_weighted_graph()
ii_edge_index, ii_weight = build_item_item_graph(train_df, k=10)

# 학습
for epoch in range(EPOCHS):
    u_emb, i_emb = model(ui_edge_index, ui_weight,
                         ii_edge_index, ii_weight)
    # BPR Loss + Hard Negative Mining (기존과 동일)
```

### 예상 효과

**Recall 향상:**
```
현재: User의 구매 이력 → 유사 User 추론 → Item 추천
V10:  User의 구매 이력 → 유사 Item 직접 추천 (shortcut)
```

**Cold Item 개선:**
- 구매 적은 item도 유사 item 정보로 embedding 개선
- Long-tail item 추천 품질 향상

**예상 성능:**
```
V9b: Recall 0.8794
V10b: Recall 0.92+ (예상)
V10b: AUC-ROC 0.93+ (예상)
```

### 구현 우선순위

1. **V10a** (V9a + Item-Item Graph)
   - 기본 검증용

2. **V10b** (V9b + Item-Item Graph) ⭐ 메인
   - 최고 성능 기대
   - Rating-aware + Item similarity

### 하이퍼파라미터

```python
K_NEIGHBORS = 10  # Item-Item graph의 이웃 수
ALPHA = 0.5       # UI/II graph mixing ratio (학습 가능)
SIMILARITY = 'cosine'  # cosine, jaccard, pearson 중 선택
```

### 실험 계획

1. **Baseline 비교**
   - V9b (without II graph)
   - V10b (with II graph)

2. **Ablation Study**
   - K=5, 10, 20 비교
   - Cosine vs Jaccard similarity
   - Fixed alpha vs Learnable alpha

3. **성능 측정**
   - AUC-ROC, F1, Precision, Recall
   - Cold item 성능 (interaction < 10)
   - Long-tail item 성능 (bottom 20%)

---

## V11 계획: Self-Supervised Contrastive Learning

### 전제 조건
- V10에서 +3% 이상 향상 확인

### 목표
- **Target AUC-ROC**: 0.94+
- **Target F1**: 0.90+
- **예상 향상**: +3-8% 전반적

### 핵심 아이디어

**문제:**
- Supervised BPR Loss만 사용
- Label이 없는 대부분의 (user, item) pair 정보 미활용

**해결:**
Self-supervised learning으로 robust representation 학습
```
같은 user/item의 다른 view끼리 유사하게
다른 user/item끼리 멀게
```

### 구현 방법

#### 1. Graph Augmentation
```python
def augment_graph(edge_index, drop_rate=0.1):
    """
    Random edge dropout으로 augmented view 생성
    """
    mask = torch.rand(edge_index.shape[1]) > drop_rate
    return edge_index[:, mask]

# Two augmented views
aug1_edge = augment_graph(edge_index, 0.1)
aug2_edge = augment_graph(edge_index, 0.1)
```

#### 2. InfoNCE Contrastive Loss
```python
def contrastive_loss(emb1, emb2, temperature=0.1):
    """
    두 view의 embedding이 일치하도록
    """
    # Normalize
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    # Positive similarity (같은 user/item)
    pos_sim = (emb1 * emb2).sum(dim=1) / temperature

    # Negative similarity (모든 pair)
    all_sim = emb1 @ emb2.T / temperature

    # InfoNCE
    loss = -torch.log(
        torch.exp(pos_sim) / torch.exp(all_sim).sum(dim=1)
    ).mean()

    return loss
```

#### 3. Multi-task Learning
```python
# Total Loss = BPR + SSL
for epoch in range(EPOCHS):
    # Supervised: BPR Loss
    u_emb, i_emb = model(edge_index, edge_weight)
    bpr = bpr_loss(pos_scores, neg_scores)

    # Self-supervised: Contrastive Loss
    u_emb1, i_emb1 = model(aug1_edge, aug1_weight)
    u_emb2, i_emb2 = model(aug2_edge, aug2_weight)
    ssl_user = contrastive_loss(u_emb1, u_emb2)
    ssl_item = contrastive_loss(i_emb1, i_emb2)

    # Combined
    total_loss = bpr + 0.1 * (ssl_user + ssl_item)
```

### 예상 효과

- **Robustness**: Noise에 강한 representation
- **Generalization**: 미관측 pair 예측 향상
- **Cold Start**: 적은 interaction으로도 좋은 embedding

---

## V12+ 미래 계획

### Option 1: Bayesian Embedding (불확실성 모델링)
```python
# User/Item embedding에 uncertainty 추가
user_emb_mean, user_emb_logvar = model.encode(user)
# Cold user → 높은 uncertainty
# Active user → 낮은 uncertainty
```

**효과:**
- Cold start problem 개선
- Calibrated prediction (신뢰도 제공)

### Option 2: Multi-Interest User Modeling
```python
# User가 여러 interest cluster 보유
user_interests = [interest_1, interest_2, interest_3, ...]
# Item마다 가장 관련있는 interest 선택
```

**효과:**
- 다양한 취향 가진 user 모델링
- 추천 다양성 향상

### Option 3: Temporal Dynamics
```python
# Time-aware embedding
user_emb = f(user_id, timestamp)
# 최근 구매에 더 큰 가중치
```

**효과:**
- 취향 변화 반영
- 계절성/트렌드 고려

---

## 실험 방법론

### 성능 평가 기준

**Keep (유지):**
- 개선이 +1% 이상
- 복잡도 증가가 합리적

**Drop (버림):**
- 개선이 +0.5% 미만
- 성능 하락
- 과도한 복잡도 증가

### 버전 네이밍 규칙
```
V{N}a: Rating 무시 (모든 구매 = positive)
V{N}b: Rating-aware (Rating >= 4만 positive)
```

### Ablation Study 원칙
```
각 breakthrough마다:
1. 단독 효과 측정
2. 조합 효과 측정 (시너지 확인)
3. 최적 하이퍼파라미터 탐색
```

---

## 목표 타임라인

### 단기 (1-2주)
- [x] V9a/V9b 완료 (Hard Negative + Rating Weighted)
- [ ] V10a/V10b 구현 (Item-Item Graph)
- [ ] V10 성능 평가 및 분석

### 중기 (3-4주)
- [ ] V11a/V11b 구현 (Contrastive Learning)
- [ ] V11 성능 평가
- [ ] 최종 모델 선택 (V10 vs V11)

### 장기 (1-2개월)
- [ ] Production deployment 준비
- [ ] Online A/B testing
- [ ] User feedback 수집

---

## 성공 지표

### 최소 목표
- AUC-ROC: 0.93+
- F1 Score: 0.88+
- Recall: 0.90+

### 도전 목표
- AUC-ROC: 0.95+
- F1 Score: 0.92+
- Top-10 Hit Rate: 0.80+

---

## 참고 문헌

### 이론적 배경
- Book: "High-Dimensional Data Analysis with Low-Dimensional Models"
  - Chapter 4: Matrix Completion (Item-Item similarity)
  - Chapter 5: Robust PCA (Noise modeling)
  - Chapter 7: Nonconvex Optimization (Hard negative)
  - Chapter 16: Rate Reduction (Contrastive learning)

### 구현 참고
- LightGCN: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
- Hard Negative Mining: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
- Self-supervised GNN: "Self-supervised Graph Learning for Recommendation"

---

## 마지막 업데이트: 2025-11-18

**현재 Best 모델:** V9b
- AUC-ROC: 0.9263
- F1: 0.8719
- Recall: 0.8794

**다음 목표:** V10b (+ Item-Item Graph)
- 예상 AUC-ROC: 0.93+
- 예상 F1: 0.88+
