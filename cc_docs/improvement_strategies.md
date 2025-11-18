# CCA1 & CCB1 개선 전략 및 Breakthrough Ideas

**작성일**: 2025-11-18
**현재 성능**: CCA1 (AUC 0.889), CCB1 (AUC 0.927)
**목표**: 0.93+ (CCA1), 0.95+ (CCB1)

---

## Part 1: 현재 모델의 한계점 분석

### CCA1 (Binary Classification)의 문제점

#### 1. 품질 미고려 (Critical Issue ⚠️)
```
문제: Rating 0.5도 positive로 취급
결과: 사용자가 싫어한 영화도 추천될 수 있음
영향: "잘못된 추천은 감점" 규칙에서 불리
```

**실제 예시**:
- User A가 영화 X를 봤지만 rating 1.0 (최악)
- CCA1: "연결이 있으니 비슷한 영화 추천" → 잘못된 추천
- CCB1: "Rating < 4니까 추천 안함" → 올바른 판단

#### 2. Precision/Recall Trade-off 한계
- Precision: 87.2% (좋음)
- Recall: 82.0% (개선 여지)
- **문제**: 좋은 영화를 18%나 놓침

#### 3. Score Distribution 문제
- Positive mean: 1.34, std: 0.80
- Negative mean: 0.16, std: 0.39
- **Gap**: 1.18 (충분하지만 CCB1의 1.15보다 작음)

#### 4. Top-K Ranking의 불안정성
- Recall@K가 epoch마다 크게 변동 (0.11 ~ 0.15)
- Best epoch 25 이후 성능 하락

---

### CCB1 (Rating Prediction)의 문제점

#### 1. Rating 정보 의존성 (Critical Issue ⚠️)
```
문제: Test set에 rating 없으면 판단 기준 상실
결과: sample2.csv에서 모두 O 예측 (너무 공격적)
영향: Real-world에서 rating 없는 경우 대응 불가
```

**해결 필요**:
- Rating 없을 때 Conservative 전략 필요
- Uncertainty estimation 필요

#### 2. Rating < 4 데이터의 비효율적 활용
- Rating < 4: Train graph에만 포함 (평가 제외)
- **문제**: 53,309개의 negative signal을 충분히 활용 못함
- **개선 가능**: Negative도 적극적으로 학습에 활용

#### 3. Good Purchase의 Imbalance
- Good (>=4): 51,830개 (49.3%)
- Poor (<4): 53,309개 (50.7%)
- **거의 균형**: 좋지만, class weight 조정으로 더 개선 가능

#### 4. Recall@K의 변동성
- 0.15 ~ 0.23 사이에서 큰 변동
- CCA1보다 더 불안정

---

## Part 2: 즉시 적용 가능한 개선 (v2 Level)

### A. CCA1 개선 방향 (cca2)

#### 1. Weighted BPR Loss (품질 반영)
**아이디어**: Rating을 loss weight로 활용
```python
# 현재 (cca1)
loss = -log(sigmoid(pos_score - neg_score))

# 개선 (cca2)
rating_weight = 0.5 + 0.1 * rating  # rating 5 -> weight 1.0
loss = rating_weight * (-log(sigmoid(pos_score - neg_score)))
```

**기대 효과**:
- 높은 rating의 interaction에 더 집중
- 낮은 rating은 덜 중요하게 학습
- 품질 고려하면서도 모든 데이터 활용
- **예상 AUC**: 0.89 → 0.91

#### 2. Score Re-calibration (Threshold 개선)
**아이디어**: Rating 분포를 고려한 threshold 조정
```python
# User별 평균 rating 계산
user_avg_rating = df.groupby('user')['rating'].mean()

# Threshold를 user별로 조정
adjusted_threshold = base_threshold * (user_avg_rating / 4.0)
```

**기대 효과**:
- 평가 기준이 높은 user: threshold 높임
- 평가 기준이 낮은 user: threshold 낮춤
- **예상 F1**: 0.845 → 0.860

#### 3. Multi-Hop Attention (Layer 개선)
**아이디어**: Layer별로 다른 가중치 학습
```python
# 현재: Simple mean
final_emb = mean([emb_0, emb_1, emb_2])

# 개선: Learnable attention
alpha = softmax([w_0, w_1, w_2])  # learnable
final_emb = alpha_0 * emb_0 + alpha_1 * emb_1 + alpha_2 * emb_2
```

**기대 효과**:
- Layer별 중요도 자동 학습
- 더 expressive한 representation
- **예상 AUC**: +0.5% ~ 1%

---

### B. CCB1 개선 방향 (ccb2)

#### 1. Uncertainty-Aware Recommendation (Rating 의존성 해결 ⭐)
**아이디어**: Bayesian 또는 Dropout으로 uncertainty 측정
```python
# MC Dropout으로 uncertainty 추정
def predict_with_uncertainty(model, user, item, n_samples=10):
    model.train()  # Enable dropout
    scores = []
    for _ in range(n_samples):
        score = model(user, item)
        scores.append(score)

    mean_score = np.mean(scores)
    uncertainty = np.std(scores)

    # High uncertainty면 conservative
    if uncertainty > threshold_unc:
        return 'X'  # Don't recommend
    else:
        return 'O' if mean_score > threshold else 'X'
```

**기대 효과**:
- Rating 없어도 모델의 확신도로 판단 가능
- Sample2 문제 해결
- **예상**: Conservative but accurate

#### 2. Contrastive Learning for Better Separation (AUC 향상 ⭐)
**아이디어**: Positive끼리 가깝게, Negative와 멀게
```python
# Contrastive Loss (SimCLR style)
def contrastive_loss(pos_emb, neg_emb, temperature=0.1):
    # Positive pairs: 가깝게
    pos_sim = cosine_similarity(pos_emb[0], pos_emb[1])

    # Negative pairs: 멀게
    neg_sim = cosine_similarity(pos_emb[0], neg_emb)

    loss = -log(exp(pos_sim / T) / (exp(pos_sim / T) + sum(exp(neg_sim / T))))
    return loss

# Total Loss
total_loss = BPR_loss + lambda_cl * contrastive_loss
```

**기대 효과**:
- Embedding space에서 더 명확한 구조
- AUC 향상 (separation 개선)
- **예상 AUC**: 0.927 → 0.940+

#### 3. Rating Distribution Modeling (Rating 정보 최대 활용)
**아이디어**: Rating을 categorical로 취급하여 분포 학습
```python
# Rating을 10-class classification
rating_logits = MLP(user_emb, item_emb)  # (10,)
rating_probs = softmax(rating_logits)

# Expected rating
expected_rating = sum(rating_probs[i] * (i * 0.5 + 0.5) for i in range(10))

# Loss
classification_loss = CrossEntropy(rating_logits, true_rating_class)
total_loss = BPR_loss + lambda_rating * classification_loss
```

**기대 효과**:
- Rating의 ordinal nature 활용
- 더 정확한 rating 예측 가능
- **예상 F1**: 0.872 → 0.885

#### 4. Negative Sampling 고도화
**아이디어**: Hard negative 비율을 동적으로 조정
```python
# Curriculum Learning: Easy → Hard
hard_ratio = min(0.8, 0.2 + epoch / total_epochs * 0.6)

# Adaptive Hard Negative: Loss 높은 것 위주로
if loss > threshold:
    hard_ratio = 0.7  # More hard negatives
else:
    hard_ratio = 0.3  # Less hard negatives
```

**기대 효과**:
- 학습 초반: Easy negative로 안정화
- 학습 후반: Hard negative로 fine-tuning
- **예상**: 수렴 속도 +20%, 성능 +0.5%

---

## Part 3: Breakthrough Ideas (v3+ Level)

### Idea 1: Self-Supervised Auxiliary Tasks (대형 개선 🚀)

#### 컨셉
**"User-item interaction 외의 신호로 representation 강화"**

#### 구체적 방법
```python
# Task 1: User-User Similarity Prediction
# 비슷한 취향의 user pair를 positive로
similar_users = find_similar_users_by_jaccard(threshold=0.3)
loss_uu = contrastive_loss(user_emb[u1], user_emb[u2], label='similar')

# Task 2: Item-Item Co-occurrence Prediction
# 같은 user가 본 item pair를 positive로
co_occur_items = find_co_occurrence_items()
loss_ii = contrastive_loss(item_emb[i1], item_emb[i2], label='co-occur')

# Task 3: Temporal Order Prediction (if timestamp available)
# User가 item1 → item2 순서로 봤으면 그 순서 학습
loss_temp = temporal_order_loss(user, item1, item2)

# Total Loss
loss = BPR_loss + λ1 * loss_uu + λ2 * loss_ii + λ3 * loss_temp
```

**기대 효과**:
- Rich representation learning
- Cold start 개선 (새 user/item도 similarity로 추론)
- **예상 AUC**: +2% ~ 3%

**난이도**: ⭐⭐⭐⭐ (구현 복잡, 효과 불확실)

---

### Idea 2: Graph Structure Learning (그래프 구조 개선 🚀)

#### 현재 문제
- Graph = Train data edges only
- **문제**: Train에 없는 latent connection 놓침

#### 개선 방법
**Learnable Edge Addition**: 유사한 user/item에 soft edge 추가
```python
# User similarity graph
user_sim = cosine_similarity(user_emb, user_emb)  # (n_users, n_users)
user_adj = (user_sim > threshold).float() * user_sim

# Item similarity graph
item_sim = cosine_similarity(item_emb, item_emb)
item_adj = (item_sim > threshold).float() * item_sim

# Augmented graph
edge_index_aug = concat([
    edge_index_original,
    user_user_edges,
    item_item_edges
])

# GNN on augmented graph
emb = LightGCN(edge_index_aug, edge_weight_aug)
```

**기대 효과**:
- Higher-order connectivity 활용
- Long-tail item 추천 개선
- **예상 AUC**: +1% ~ 2%

**난이도**: ⭐⭐⭐⭐ (graph 크기 증가, 메모리 이슈)

---

### Idea 3: Multi-Task Learning (A + B 동시 학습 🚀)

#### 컨셉
**"Binary classification과 Rating prediction을 동시에 학습"**

#### 구현
```python
class MultiTaskGNN(nn.Module):
    def __init__(self):
        self.shared_gnn = LightGCN(...)
        self.binary_head = MLP(emb_dim, 1)  # Binary
        self.rating_head = MLP(emb_dim, 1)  # Rating

    def forward(self, user, item):
        u_emb, i_emb = self.shared_gnn(edge_index, edge_weight)

        # Task 1: Binary
        binary_score = self.binary_head(u_emb[user] * i_emb[item])

        # Task 2: Rating
        rating_pred = self.rating_head(u_emb[user] * i_emb[item])

        return binary_score, rating_pred

# Loss
loss_binary = BPR_loss(binary_score_pos, binary_score_neg)
loss_rating = MSE_loss(rating_pred, rating_true)
total_loss = loss_binary + λ * loss_rating
```

**기대 효과**:
- Shared representation이 더 강건해짐
- Binary와 Rating의 상호 보완
- **예상**: 두 task 모두 +1% ~ 2%

**장점**: 하나의 모델로 A/B 두 전략 커버 가능

**난이도**: ⭐⭐⭐ (구현 중간, 효과 높음)

---

### Idea 4: Attention-based GNN (표현력 강화 🚀)

#### 현재 LightGCN의 한계
- 모든 neighbor를 동등하게 취급 (degree로만 normalize)
- **문제**: 중요한 neighbor와 덜 중요한 neighbor 구분 못함

#### GAT (Graph Attention Network) 도입
```python
class LightGAT(nn.Module):
    def __init__(self):
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.attn = nn.Linear(emb_dim * 2, 1)  # Attention

    def forward(self, edge_index, edge_weight):
        all_emb = concat([user_emb.weight, item_emb.weight])

        for layer in range(n_layers):
            # Compute attention weights
            row, col = edge_index
            alpha = softmax(self.attn(concat([all_emb[row], all_emb[col]])))

            # Weighted aggregation
            messages = all_emb[col] * alpha * edge_weight
            all_emb = scatter_add(messages, row, dim=0)

        return all_emb[:n_users], all_emb[n_users:]
```

**기대 효과**:
- Important neighbor에 더 집중
- More expressive representation
- **예상 AUC**: +1% ~ 2%

**단점**:
- 학습 느림 (attention 계산)
- Overfitting 위험
- **난이도**: ⭐⭐⭐⭐

---

### Idea 5: Curriculum Learning (학습 전략 개선 🚀)

#### 컨셉
**"Easy examples → Hard examples 순서로 학습"**

#### 구현
```python
# Step 1: Easy/Hard 정의
def get_difficulty(user, item, rating):
    # Easy: Rating 극단값 (0.5 또는 5.0)
    # Hard: Rating 중간값 (3.0, 3.5)
    return abs(rating - 3.0)

# Step 2: Curriculum schedule
def get_samples_for_epoch(epoch, total_epochs):
    if epoch < total_epochs * 0.3:
        # 초반: Easy만
        return df[df['difficulty'] > 1.5]
    elif epoch < total_epochs * 0.6:
        # 중반: Easy + Medium
        return df[df['difficulty'] > 0.5]
    else:
        # 후반: All
        return df
```

**기대 효과**:
- 학습 초반 안정화
- Hard case 성능 향상
- **예상**: F1 +0.5% ~ 1%

**난이도**: ⭐⭐ (구현 쉬움, 효과 중간)

---

### Idea 6: Ensemble with Diversity (앙상블 강화 🚀)

#### 현재 계획
- CCA1 + CCB1 단순 averaging

#### 개선: Diversity 확보
```python
# Model 1: CCA1 (emb_dim=32, layers=2)
# Model 2: CCB1 (emb_dim=32, layers=2)
# Model 3: CCA1 (emb_dim=64, layers=1)  # Different architecture
# Model 4: CCB1 (emb_dim=16, layers=3)  # Different architecture
# Model 5: GAT-based (emb_dim=32, layers=2, attention)

# Weighted ensemble
weights = [0.15, 0.30, 0.15, 0.25, 0.15]  # CCB1 highest
final_score = sum(w * model.predict() for w, model in zip(weights, models))
```

**기대 효과**:
- Error의 independence 증가 → ensemble 효과 극대화
- **예상 AUC**: +1% ~ 3%

**난이도**: ⭐⭐ (모델 여러개 학습 필요)

---

### Idea 7: Cold Start Enhancement (실전 대비 🚀)

#### 문제 인식
- 현재: 모든 user가 충분한 데이터 (>10 interactions)
- **실전**: 새 user/item 등장 가능

#### 해결 방법
**Meta-Learning (MAML) 적용**
```python
# Few-shot learning for new users
def adapt_to_new_user(new_user_interactions):
    # Step 1: Initialize with global model
    user_emb_init = model.user_emb.mean(dim=0)

    # Step 2: Few-shot adaptation (1-5 interactions)
    for _ in range(5):  # Inner loop
        loss = compute_loss(new_user_interactions)
        user_emb_new = user_emb_init - lr * grad(loss)

    return user_emb_new
```

**기대 효과**:
- 새 user도 빠르게 적응
- 실전 robustness 증가

**난이도**: ⭐⭐⭐⭐⭐ (매우 어려움, 우리 데이터에선 불필요)

---

### Idea 8: Rating Calibration with Debiasing (편향 제거 🚀)

#### 관찰
- User마다 rating 평균이 다름
  - User A: 평균 4.5 (관대한 평가)
  - User B: 평균 2.5 (엄격한 평가)
- **문제**: Rating 4가 user마다 다른 의미

#### 해결: Debiasing
```python
# User bias 계산
user_bias = df.groupby('user')['rating'].mean() - df['rating'].mean()

# Rating 정규화
df['rating_normalized'] = df['rating'] - df['user'].map(user_bias)

# 학습 시 normalized rating 사용
edge_weight = 0.4 + 0.15 * rating_normalized
```

**기대 효과**:
- User간 공정한 비교
- Rating weighting 정확도 향상
- **예상 AUC**: +0.5% ~ 1%

**난이도**: ⭐⭐ (쉬움)

---

## Part 4: 우선순위 및 실행 계획

### Tier 1 (즉시 실행, 높은 효과/낮은 난이도) ⭐⭐⭐

#### CCA2
1. **Weighted BPR Loss** (Rating weight 반영)
   - 예상 효과: AUC +2% (0.89 → 0.91)
   - 난이도: ⭐⭐
   - 시간: 1시간

2. **Rating Calibration** (User bias 제거)
   - 예상 효과: AUC +0.5%
   - 난이도: ⭐
   - 시간: 30분

#### CCB2
1. **Contrastive Learning** (Separation 강화)
   - 예상 효과: AUC +1% (0.927 → 0.937)
   - 난이도: ⭐⭐⭐
   - 시간: 2시간

2. **Uncertainty-Aware Prediction** (Sample2 문제 해결)
   - 예상 효과: Conservative but accurate
   - 난이도: ⭐⭐
   - 시간: 1시간

**예상 결과**:
- CCA2: AUC 0.915 (현재 0.889)
- CCB2: AUC 0.940 (현재 0.927)

---

### Tier 2 (중기, 중간 효과/중간 난이도) ⭐⭐

#### CCA3 / CCB3
1. **Multi-Task Learning** (Binary + Rating)
   - 예상 효과: +1% ~ 2%
   - 난이도: ⭐⭐⭐
   - 시간: 3-4시간

2. **Curriculum Learning**
   - 예상 효과: +0.5% ~ 1%
   - 난이도: ⭐⭐
   - 시간: 1시간

3. **Dynamic Hard Negative Sampling**
   - 예상 효과: 수렴 속도 +20%
   - 난이도: ⭐⭐
   - 시간: 1시간

**예상 결과**:
- CCA3: AUC 0.925
- CCB3: AUC 0.950

---

### Tier 3 (장기, 연구 수준) ⭐

1. **Graph Attention (GAT)**
   - 예상 효과: +1% ~ 2%
   - 난이도: ⭐⭐⭐⭐
   - 시간: 5-6시간

2. **Self-Supervised Learning**
   - 예상 효과: +2% ~ 3%
   - 난이도: ⭐⭐⭐⭐
   - 시간: 6-8시간

3. **Graph Structure Learning**
   - 예상 효과: +1% ~ 2%
   - 난이도: ⭐⭐⭐⭐⭐
   - 시간: 8-10시간

---

## Part 5: 최종 권장 로드맵

### Phase 1: Quick Wins (v2) - 1일
```
CCA2:
✓ Weighted BPR Loss
✓ Rating Calibration
→ 목표: AUC 0.91

CCB2:
✓ Contrastive Learning
✓ Uncertainty Estimation
→ 목표: AUC 0.94
```

### Phase 2: Strategic Improvements (v3) - 2일
```
✓ Multi-Task Learning (A+B 통합)
✓ Curriculum Learning
✓ Advanced Negative Sampling
→ 목표: AUC 0.93 (A), 0.95 (B)
```

### Phase 3: Research-Level (v4+) - 선택적
```
✓ GAT
✓ Self-Supervised
✓ Graph Structure Learning
→ 목표: SOTA (0.96+)
```

---

## Part 6: 평가 규칙 대응 전략

### "잘못된 추천은 감점" 대응

#### Conservative 전략
```python
# Precision 우선 threshold
threshold_conservative = optimal_threshold * 1.15

# Uncertainty 기반 필터링
if uncertainty > 0.3:
    return 'X'  # Don't recommend if uncertain

# Top-K 제한
K_conservative = max(2, int(K_optimal * 0.8))
```

#### Ensemble with Voting
```python
# 여러 모델이 모두 O일 때만 O
votes = [model1.predict(), model2.predict(), model3.predict()]
if votes.count('O') >= 3:
    return 'O'
else:
    return 'X'
```

---

## Part 7: 실험 체크리스트

### CCA2 실험
- [ ] Weighted BPR Loss 구현
- [ ] Rating Calibration 구현
- [ ] Baseline (CCA1) 대비 성능 비교
- [ ] Sample 테스트 (형식 포함)
- [ ] Threshold 재튜닝
- [ ] 문서화

### CCB2 실험
- [ ] Contrastive Loss 구현
- [ ] MC Dropout Uncertainty 구현
- [ ] Baseline (CCB1) 대비 성능 비교
- [ ] Sample2 문제 해결 확인
- [ ] Conservative mode 테스트
- [ ] 문서화

### 비교 분석
- [ ] CCA1 vs CCA2
- [ ] CCB1 vs CCB2
- [ ] CCA2 vs CCB2 (최종 선택)
- [ ] Ensemble 시도
- [ ] 결과 시각화

---

## 결론

### 핵심 전략
1. **Tier 1 먼저 구현** (Weighted BPR, Contrastive, Uncertainty)
2. **CCB2에 집중** (이미 CCB1이 우수하므로)
3. **Conservative 전략 준비** (평가 규칙 대응)
4. **Ensemble 준비** (여러 모델 학습)

### 예상 최종 성능
- **CCA2**: AUC 0.91, F1 0.86
- **CCB2**: AUC 0.94, F1 0.88
- **Ensemble**: AUC 0.95+, F1 0.89+

### 성공 가능성
- Tier 1: 90% (검증된 방법)
- Tier 2: 70% (효과 불확실)
- Tier 3: 50% (연구 수준)

**추천**: Tier 1 → Tier 2 순서로 진행, Tier 3는 시간 여유 있을 때만

---

**문서 버전**: v1.0
**최종 수정**: 2025-11-18
