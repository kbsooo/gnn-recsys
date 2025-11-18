# CCA & CCB 개선 전략 v2 (수정판)

**작성일**: 2025-11-18
**현재 성능**: CCA1 (AUC 0.889), CCB1 (AUC 0.927)
**목표**: CCA2 (AUC 0.91+), CCB2 (AUC 0.94+)

---

## Part 1: 모델 기본 원리 재정립

### CCA (Binary Classification Approach)

#### 핵심 원리
```
학습: User-Item edge 존재 = Positive (rating 완전 무시)
테스트: User-Item 쌍만 주어짐
예측: 해당 쌍이 연결될 확률 (구매 예측)
추천: P(connection) > threshold → O
```

#### 특징
- ✅ Rating 정보 완전히 무시
- ✅ 구매 행위 자체에 의미
- ✅ "이 사용자가 이 아이템을 구매할까?" 예측
- ⚠️ 낮은 평점 구매도 positive → 품질 미고려

---

### CCB (Rating Prediction Approach)

#### 핵심 원리
```
학습: User-Item 관계 + Rating 값 학습
테스트: User-Item 쌍만 주어짐 (rating column 없음!)
예측: 해당 쌍의 Rating 값 예측 (regression)
추천: Predicted_Rating >= 4.0 → O (고품질만 추천)
```

#### 특징
- ✅ Rating 값 자체를 예측 (regression task)
- ✅ 품질 있는 추천 (예측 rating >= 4만 추천)
- ✅ "이 사용자가 이 아이템에 몇 점을 줄까?" 예측
- ⚠️ Test set에는 rating column이 없어야 함

---

## Part 2: 현재 CCB1의 문제점 분석

### 문제 1: Rating 값을 예측하지 않음 (Critical ⚠️)

**현재 CCB1 구현**:
```python
# BPR Loss (ranking-based)
pos_score = (u_emb[user] * i_emb[item]).sum()  # ← 이건 그냥 score
neg_score = (u_emb[user] * i_emb[neg_item]).sum()
loss = -log(sigmoid(pos_score - neg_score))

# Inference
score = (u_emb[user] * i_emb[item]).sum()
if score > 0.4909:  # ← score를 threshold와 비교
    return 'O'
```

**문제점**:
- `score`는 임베딩 내적값일 뿐, **rating이 아님**
- Score 범위: 보통 -2 ~ +3 정도 (rating 0.5~5와 무관)
- Threshold 0.4909는 "score > 0.4909"이지, "rating >= 4"가 아님

**결과**:
- CCB1은 사실상 CCA1과 거의 동일한 방식 (rating weighting만 다름)
- Rating 값을 직접 예측하지 않음

---

### 문제 2: Sample1.csv에서 rating column 사용

**현재 구현**:
```python
if rating is not None and rating < GOOD_RATING_THRESHOLD:
    return 'X'  # Rating < 4면 무조건 X
```

**문제**:
- Test set에 rating이 있다고 가정
- 실전에서는 rating column이 없음
- 모델의 예측 능력을 사용하지 않음

---

## Part 3: CCB2 개선 방안 (핵심)

### 목표: Rating Regression 모듈 추가

CCB2는 **Rating 값 자체를 예측**해야 함.

---

### 개선 1: Rating Prediction Head 추가 ⭐⭐⭐⭐⭐

#### 구현

**Step 1: 모델에 Rating Regression Head 추가**
```python
class LightGCN_with_Rating(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, n_layers=2):
        super().__init__()
        # 기존 LightGCN
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.n_layers = n_layers

        # ★ Rating Prediction Head 추가
        self.rating_mlp = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, edge_index, edge_weight):
        # LightGCN message passing
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]

        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = scatter_add(messages, row, ...)
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs), dim=0)
        u_emb = final_emb[:n_users]
        i_emb = final_emb[n_users:]

        return u_emb, i_emb

    def predict_rating(self, user_idx, item_idx):
        """Rating 값 예측 (0.5 ~ 5.0)"""
        u_emb, i_emb = self.forward(edge_index, edge_weight)

        # Element-wise product
        interaction = u_emb[user_idx] * i_emb[item_idx]

        # MLP로 rating 예측
        rating_logit = self.rating_mlp(interaction).squeeze()

        # Sigmoid로 0~1 → 0.5~5.0으로 scaling
        rating = torch.sigmoid(rating_logit) * 4.5 + 0.5

        return rating
```

**Step 2: Hybrid Loss (BPR + MSE)**
```python
def hybrid_loss(model, pos_u, pos_i, pos_rating, neg_i, lambda_mse=0.5):
    """
    BPR Loss: Ranking 학습
    MSE Loss: Rating 값 학습
    """
    u_emb, i_emb = model(edge_index, edge_weight)

    # BPR Loss (Ranking)
    pos_score = (u_emb[pos_u] * i_emb[pos_i]).sum(dim=1)
    neg_score = (u_emb[pos_u].unsqueeze(1) * i_emb[neg_i]).sum(dim=2)
    bpr = -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-8).mean()

    # MSE Loss (Rating Prediction)
    pred_rating = model.predict_rating(pos_u, pos_i)
    mse = F.mse_loss(pred_rating, pos_rating)

    # Total Loss
    total_loss = bpr + lambda_mse * mse

    return total_loss, bpr.item(), mse.item()
```

**Step 3: Inference (Rating 기반 추천)**
```python
def predict_hybrid_ccb2(test_input_df, rating_threshold=4.0):
    """
    Rating 값을 예측하고, >= 4.0이면 추천
    """
    model.eval()
    results = []
    stats = {'total_o': 0, 'total_items': 0}

    with torch.no_grad():
        for _, row in test_input_df.iterrows():
            user = row['user']
            item = row['item']

            # Unknown user/item → X
            if user not in user2idx or item not in item2idx:
                results.append({'user': user, 'item': item, 'recommend': 'X'})
                stats['total_items'] += 1
                continue

            user_idx = user2idx[user]
            item_idx = item2idx[item]

            # Train에 있는 item → X
            if item_idx in user_train_items[user_idx]:
                results.append({'user': user, 'item': item, 'recommend': 'X'})
                stats['total_items'] += 1
                continue

            # ★ Rating 예측
            pred_rating = model.predict_rating(
                torch.tensor([user_idx]).to(device),
                torch.tensor([item_idx]).to(device)
            ).item()

            # ★ Rating >= 4.0이면 추천
            if pred_rating >= rating_threshold:
                recommend = 'O'
                stats['total_o'] += 1
            else:
                recommend = 'X'

            results.append({
                'user': user,
                'item': item,
                'recommend': recommend,
                'predicted_rating': pred_rating  # Optional: 디버깅용
            })
            stats['total_items'] += 1

    return pd.DataFrame(results), stats
```

**기대 효과**:
- ✅ Rating 값 직접 예측 (0.5 ~ 5.0)
- ✅ Test set에 rating column 불필요
- ✅ 품질 기반 추천 (pred_rating >= 4)
- ✅ 해석 가능성 높음 ("이 아이템은 4.2점 예상")
- **예상 AUC**: 0.93 ~ 0.95

---

### 개선 2: Rating Distribution Modeling (Advanced)

#### Ordinal Regression Approach

Rating을 10개 클래스로 취급 (0.5, 1.0, ..., 5.0)

```python
class RatingClassifier(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.classifier = nn.Linear(emb_dim, 10)  # 10 classes

    def forward(self, interaction_emb):
        logits = self.classifier(interaction_emb)
        return logits

# Loss
def ordinal_loss(logits, true_rating_class):
    """Cross-entropy for ordinal regression"""
    return F.cross_entropy(logits, true_rating_class)

# Prediction
def predict_rating(logits):
    probs = F.softmax(logits, dim=-1)
    # Expected rating
    rating = sum(probs[i] * (i * 0.5 + 0.5) for i in range(10))
    return rating
```

**장점**:
- Rating의 ordinal nature 활용
- 더 정확한 예측 가능

**단점**:
- 구현 복잡도 증가

**예상 성능**: AUC +0.5% ~ 1%

---

## Part 4: CCA2 개선 방안

CCA는 rating을 무시하므로, 기존 방향 유지.

### 개선 1: Weighted BPR Loss (Rating을 Weight로만 사용)

```python
def weighted_bpr_loss(pos_scores, neg_scores, pos_ratings):
    """
    Rating이 높은 edge를 더 중요하게 학습
    (Rating 값 자체를 예측하는 건 아님)
    """
    # Rating → Weight
    weights = 0.5 + 0.1 * pos_ratings  # rating 5 → weight 1.0

    # Weighted BPR
    diff = pos_scores.unsqueeze(1) - neg_scores
    loss = -torch.log(torch.sigmoid(diff) + 1e-8)
    weighted_loss = (loss * weights.unsqueeze(1)).mean()

    return weighted_loss
```

**효과**:
- 높은 rating edge를 더 학습
- 낮은 rating edge는 덜 중요하게
- **Rating 예측하는 건 아님** (weight로만 사용)

**예상 AUC**: 0.89 → 0.91

---

### 개선 2: User Bias Calibration

```python
# User별 평균 rating 계산
user_avg_rating = df.groupby('user')['rating'].mean()

# Rating 정규화 (user bias 제거)
df['rating_normalized'] = df['rating'] - df['user'].map(user_avg_rating) + df['rating'].mean()

# Weighted BPR에서 normalized rating 사용
weights = 0.5 + 0.1 * rating_normalized
```

**효과**: User간 공정한 비교

**예상 AUC**: +0.5%

---

## Part 5: 우선순위 및 실행 계획

### Phase 1: CCB2 핵심 개선 (최우선) ⭐⭐⭐⭐⭐

**목표**: Rating Prediction 모듈 추가

**작업 내용**:
1. ✅ LightGCN에 Rating MLP Head 추가
2. ✅ Hybrid Loss (BPR + MSE) 구현
3. ✅ Inference 함수 수정 (rating 예측 기반)
4. ✅ Sample1/Sample2 테스트 (rating column 제거)

**예상 시간**: 2-3시간

**예상 성능**:
- AUC: 0.927 → **0.94**
- Rating RMSE: **0.5 ~ 0.7** (new metric)
- 추천 품질 대폭 향상

---

### Phase 2: CCA2 개선 (후순위)

**목표**: Weighted BPR + Calibration

**작업 내용**:
1. ✅ Weighted BPR Loss 구현
2. ✅ User Bias Calibration
3. ✅ Threshold 재튜닝

**예상 시간**: 1-2시간

**예상 성능**:
- AUC: 0.889 → **0.91**

---

### Phase 3: Advanced (선택적)

**CCB3 옵션**:
- Ordinal Regression (Rating classification)
- Uncertainty Estimation (Bayesian)
- Contrastive Learning

**CCA3 옵션**:
- Multi-hop attention
- Curriculum Learning

---

## Part 6: CCB1 vs CCB2 비교

| 측면 | CCB1 (현재) | CCB2 (개선) |
|-----|------------|------------|
| **Loss** | BPR only | BPR + MSE |
| **예측 대상** | Score (의미 모호) | Rating 값 (0.5~5.0) |
| **추천 기준** | Score > 0.49 | Rating >= 4.0 |
| **Rating 활용** | Edge weight만 | 직접 예측 |
| **Test rating** | 필요 (문제!) | 불필요 ✅ |
| **해석 가능성** | 낮음 | 높음 ✅ |
| **예상 AUC** | 0.927 | 0.94 |

---

## Part 7: 구현 체크리스트

### CCB2 구현 (ccb2.ipynb)

#### Data Preprocessing
- [ ] Train data에 rating 값 포함
- [ ] Rating을 tensor로 변환
- [ ] Train/Val/Test split 동일

#### Model
- [ ] `LightGCN_with_Rating` 클래스 구현
- [ ] Rating MLP Head 추가
- [ ] `predict_rating()` 메서드 구현

#### Training
- [ ] Hybrid Loss 함수 구현 (BPR + MSE)
- [ ] Lambda 하이퍼파라미터 튜닝 (0.3 ~ 0.7)
- [ ] Loss 분해 로깅 (BPR / MSE 따로)
- [ ] Validation: Rating RMSE 계산

#### Evaluation
- [ ] AUC-ROC (기존)
- [ ] Rating RMSE (new)
- [ ] Rating MAE (new)
- [ ] F1, Precision, Recall (기존)

#### Inference
- [ ] `predict_hybrid_ccb2()` 함수 구현
- [ ] Rating >= 4.0 기준으로 O/X
- [ ] Sample1/Sample2에서 rating column 제거 후 테스트
- [ ] AGENTS.md 형식 출력

#### Comparison
- [ ] CCB1 vs CCB2 성능 비교
- [ ] Rating 예측 정확도 분석
- [ ] Sample 케이스 비교 (predicted rating 출력)

---

### CCA2 구현 (cca2.ipynb)

#### Model
- [ ] Weighted BPR Loss 구현
- [ ] User Bias Calibration

#### Training
- [ ] Weighted loss로 학습
- [ ] Threshold 재튜닝

#### Evaluation
- [ ] CCA1 vs CCA2 비교

---

## Part 8: 예상 최종 성능

### CCB2 (Rating Prediction)
```
AUC-ROC: 0.94
Rating RMSE: 0.6
F1 Score: 0.88
Precision: 0.87
Recall: 0.89

Sample1 예측:
user  item   predicted_rating  recommend
109   3745   4.2               O
88    4447   3.1               X
71    4306   4.8               O
66    1747   2.9               X
15    66934  4.5               O
```

### CCA2 (Binary Classification)
```
AUC-ROC: 0.91
F1 Score: 0.86
Precision: 0.88
Recall: 0.84

Sample1 예측:
user  item   score   recommend
109   3745   0.82    O
88    4447   0.71    O
71    4306   0.95    O
66    1747   0.68    O
15    66934  0.88    O
```

---

## Part 9: 핵심 요약

### CCA의 본질
- **구매 행위 예측**: "이 사람이 이 아이템을 살까?"
- Rating 무시, Edge 존재만 중요
- Binary classification

### CCB의 본질
- **Rating 값 예측**: "이 사람이 이 아이템에 몇 점을 줄까?"
- Rating >= 4이면 고품질 추천
- **Regression task** (핵심!)

### CCB2의 핵심 개선
```python
# Before (CCB1)
score = u_emb * i_emb  # ← 의미 모호한 score
if score > threshold:
    return 'O'

# After (CCB2)
predicted_rating = model.predict_rating(u, i)  # ← 명확한 rating 예측
if predicted_rating >= 4.0:
    return 'O'
```

---

## 결론

### 최우선 과제: CCB2 Rating Prediction 구현

**이유**:
1. CCB1은 **Rating을 예측하지 않음** (치명적 문제)
2. Test set은 **rating column이 없어야 함**
3. Rating 예측 모듈 추가로 **근본적 개선**

**예상 효과**:
- CCB2 AUC: **0.94** (현재 0.927)
- Rating RMSE: **0.6** (새 지표)
- 해석 가능성 및 실용성 대폭 향상

### 다음 단계
1. **ccb2.ipynb 작성** (Rating Prediction Head 추가)
2. Hybrid Loss 학습
3. CCA2 개선 (선택적)

---

**문서 버전**: v2.0 (수정판)
**최종 수정**: 2025-11-18
**핵심 변경**: CCB는 Rating Regression이 본질임을 명확히 함
