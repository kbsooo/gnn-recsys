# GNN 기반 영화 추천 시스템 전략 문서

## 📋 Executive Summary

**프로젝트 목표:** 668명 사용자와 10,321개 영화의 상호작용 데이터를 기반으로 GNN을 활용한 추천 시스템 구축

**핵심 도전과제:**
- 극심한 데이터 희소성 (1.5% density)
- 롱테일 분포 (영화의 70%가 5회 이하 시청)
- 사용자별 상호작용 편차 (20~5,672개)
- Cold-start 문제

**추천 접근법:** LightGCN + BPR Loss + Negative Sampling 전략

---

## 1. 데이터 분석 및 특성

### 1.1 기본 통계
```
총 상호작용: 105,139개
사용자 수: 668명
영화 수: 10,321개
희소도: ~1.5% (가능한 조합의 98.5%가 빈 공간)

사용자당 평균 상호작용: 157개 (중앙값: 70개)
영화당 평균 상호작용: 10개 (중앙값: 3개)
```

### 1.2 데이터 분포 특성

**Rating 분포 (긍정 편향):**
- 4.0점: 27.4% (가장 많음)
- 3.0점: 20.6%
- 5.0점: 14.1%
- ≥3.5점: 약 64% (긍정적 평가 비중 높음)

**문제점:**
1. **파워유저 존재**: 최대 5,672개 시청 (평균의 36배)
2. **롱테일 영화**: 7,200개 이상의 영화가 5회 이하만 시청
3. **희소성**: 대부분의 user-item 조합이 관측되지 않음

### 1.3 그래프 구조 관점

```
Bipartite Graph:
[User Nodes] ←─── edges ───→ [Item Nodes]
    668개                        10,321개
           105,139개 edges
```

**그래프 특성:**
- 평균 user degree: 157
- 평균 item degree: 10
- 매우 불균형한 degree distribution
- 많은 leaf nodes (연결이 1~2개인 영화)

---

## 2. 문제 정의 및 목표 설정

### 2.1 Task 정의

**Primary Task:** Binary Classification (추천 O/X)
- Input: (user_id, item_id) 쌍
- Output: {O, X} (추천 여부)

**Secondary Task:** Rating Prediction (학습 중)
- Rating 정보를 버리지 않고 학습에 활용
- 최종 추론 시에만 binary로 변환

### 2.2 Threshold 전략

```python
# 제안 1: Rating 기반 (추천)
if predicted_rating >= 3.5:
    recommend = 'O'
else:
    recommend = 'X'

# 제안 2: Score 기반
if model_score >= threshold:  # threshold는 validation으로 결정
    recommend = 'O'
```

**근거:** 현재 데이터의 64%가 3.5점 이상 → 이를 "추천할 만함"으로 간주

### 2.3 평가 목표

**우선순위:**
1. **Recall@10** ≥ 0.30 (사용자가 좋아할 만한 영화를 놓치지 않기)
2. **NDCG@10** ≥ 0.35 (상위 추천의 품질)
3. **Coverage** ≥ 30% (다양한 영화 추천)

---

## 3. 데이터 전처리 파이프라인

### 3.1 Train/Validation/Test Split

**전략 A: Random Split (추천)**
```python
각 사용자별로:
- Train: 80% (상호작용 랜덤 선택)
- Validation: 10%
- Test: 10%

최소 상호작용 보장: 각 set에 최소 1개 이상
```

**전략 B: Temporal Split** (timestamp 있다면)
```python
시간순 정렬 후:
- Train: 첫 80%
- Validation: 다음 10%
- Test: 마지막 10%
```

### 3.2 Negative Sampling

**중요성:** GNN 학습에 negative examples 필수 (사용자가 안 본 영화)

**방법 1: Random Negative Sampling**
```python
각 positive interaction마다:
- 사용자가 상호작용하지 않은 영화 중 랜덤 샘플링
- 비율: positive 1 : negative 3~4
```

**방법 2: Hard Negative Sampling** (고급)
```python
- 인기있지만 사용자가 안 본 영화 (더 어려운 negative)
- Epoch마다 재샘플링 (dynamic negative sampling)
```

**추천:** Random으로 시작, 성능 정체되면 Hard로 전환

### 3.3 전처리 순서

```python
1. 결측치 확인 (없음, OK)
2. User/Item ID 연속적으로 re-indexing (0부터 시작)
3. Rating normalization (선택사항):
   - Per-user z-score normalization
   - 또는 Min-Max scaling [0, 1]
4. 극단치 처리:
   - 5,000개 이상 상호작용 사용자 → 샘플링 or 가중치 감소
   - 1~2회 시청 영화 → 제거 검토
5. Train/Val/Test split
6. Negative sampling dataset 생성
7. Graph 구축 (edge list 형태)
```

### 3.4 데이터 증강 (선택)

**Graph Augmentation:**
1. **Item-Item Similarity Edges**
   - 같이 자주 시청된 영화끼리 연결
   - Jaccard similarity > threshold인 경우

2. **User-User Similarity Edges**
   - 코사인 유사도 기반 사용자 연결
   - Top-K similar users만 연결

**주의:** 계산 비용 vs 성능 향상 trade-off 고려

---

## 4. 모델 아키텍처

### 4.1 LightGCN 선택 근거

**장점:**
1. 단순성: Unnecessary transformation 제거
2. 효율성: 빠른 학습 및 추론
3. 효과성: 추천 시스템 벤치마크에서 우수한 성능
4. 해석가능성: Layer별 정보 전파 추적 가능

**LightGCN 핵심 원리:**
```
1. User와 Item을 embedding space에 표현
2. Graph convolution을 통해 이웃 정보 집계
3. Multiple layers로 multi-hop neighborhood 정보 수집
4. Layer-wise embeddings를 weighted sum
```

### 4.2 아키텍처 상세

```python
class LightGCN:
    Input:
        - User-Item bipartite graph (edge_index)
        - User embedding matrix: [num_users, embedding_dim]
        - Item embedding matrix: [num_items, embedding_dim]
    
    Layers:
        - L layers of graph convolution (L=2 or 3)
        - Layer l: e^(l) = Aggregate(e^(l-1), neighbors)
    
    Output:
        - Final embedding: weighted average of all layers
        - e_final = α₀*e⁰ + α₁*e¹ + α₂*e² + ...
        - 일반적으로 α = 1/(L+1) (uniform weight)
    
    Prediction:
        - score(u, i) = embedding_user[u]ᵀ · embedding_item[i]
```

### 4.3 하이퍼파라미터

**초기 설정:**
```python
embedding_dim = 64        # 32는 too small, 128은 overfitting 위험
num_layers = 3            # 2~3이 적절 (너무 많으면 over-smoothing)
learning_rate = 1e-3      # Adam optimizer
batch_size = 1024         # Large batch (negative sampling 때문)
dropout = 0.1             # Light regularization
weight_decay = 1e-4       # L2 regularization
```

**탐색 범위:**
- embedding_dim: [32, 64, 128, 256]
- num_layers: [2, 3, 4]
- learning_rate: [1e-4, 5e-4, 1e-3]
- negative_ratio: [2, 3, 4, 5]

### 4.4 대안 모델 (비교군)

**1. NGCF (Neural Graph Collaborative Filtering)**
- LightGCN보다 복잡 (feature transformation 포함)
- 이론적으로 더 표현력이 높지만 실제론 LightGCN과 비슷하거나 못함

**2. GraphSAGE**
- Inductive learning 가능 (새 노드 처리)
- Node feature 필요 (여기선 없음)

**3. GAT (Graph Attention)**
- Attention mechanism으로 중요한 이웃 가중치
- 계산 비용 높고, 작은 데이터셋에선 이점 적음

**추천 전략:** LightGCN 먼저, 성능 부족하면 NGCF 시도

---

## 5. 손실 함수 설계

### 5.1 BPR Loss (Bayesian Personalized Ranking) - 주 추천

**수식:**
```
L_BPR = -Σ ln σ(ŷ_ui - ŷ_uj)

where:
- ŷ_ui: positive item i에 대한 예측 점수
- ŷ_uj: negative item j에 대한 예측 점수
- σ: sigmoid function
```

**장점:**
- Ranking 최적화에 직접적
- Implicit feedback에 적합
- Pairwise 비교로 안정적 학습

**구현:**
```python
def bpr_loss(pos_scores, neg_scores):
    # pos_scores: [batch_size, 1]
    # neg_scores: [batch_size, num_negatives]
    
    diff = pos_scores - neg_scores  # Broadcasting
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss
```

### 5.2 Weighted MSE Loss (대안 1)

**수식:**
```
L_MSE = Σ w_ui · (r_ui - ŷ_ui)²

where:
- w_ui = r_ui (rating 값 자체를 weight로)
- 또는 w_ui = 1 + α·r_ui
```

**장점:**
- Rating 정보의 granularity 보존
- 높은 rating에 더 집중

**단점:**
- Regression task가 되어 ranking 최적화에 간접적

### 5.3 Multi-Task Loss (고급)

**결합 전략:**
```python
L_total = λ₁·L_BPR + λ₂·L_MSE + λ₃·L_reg

where:
- L_BPR: Ranking loss
- L_MSE: Rating prediction loss
- L_reg: Regularization (L2)
- λ: 가중치 (λ₁=1.0, λ₂=0.5, λ₃=1e-4)
```

**효과:** 두 목표를 동시에 최적화

### 5.4 최종 추천

**Phase 1:** BPR Loss만 사용 (단순성)
**Phase 2:** 성능 plateau되면 Multi-Task Loss 시도

---

## 6. 학습 전략

### 6.1 Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    
    # 1. Mini-batch sampling
    for batch in dataloader:
        users, pos_items, neg_items = batch
        
        # 2. Forward pass
        pos_scores = model(users, pos_items)
        neg_scores = model(users, neg_items)
        
        # 3. Compute loss
        loss = bpr_loss(pos_scores, neg_scores)
        loss += weight_decay * model.get_l2_reg()
        
        # 4. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 5. Validation
    if epoch % 5 == 0:
        val_recall = evaluate(model, val_data)
        if val_recall > best_recall:
            save_model(model, 'best_model.pt')
            best_recall = val_recall
        
        # Early stopping check
        if no_improvement_for_N_epochs:
            break
```

### 6.2 최적화 설정

**Optimizer:** Adam (adaptive learning rate)
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

**Learning Rate Scheduler:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10,
    verbose=True
)
```

**Early Stopping:**
- Patience: 20 epochs
- Monitor: Validation Recall@10

### 6.3 배치 구성

```python
Batch = {
    'users': [u1, u2, ..., u_B],
    'pos_items': [i_p1, i_p2, ..., i_pB],
    'neg_items': [
        [i_n11, i_n12, ..., i_n1K],
        [i_n21, i_n22, ..., i_n2K],
        ...
    ]  # K개 negative samples per user
}

Batch size B = 1024
Negative ratio K = 4
```

---

## 7. 평가 전략

### 7.1 평가 지표

**Recall@K** (최우선)
```
Recall@K = |Recommended ∩ Relevant| / |Relevant|

K = 10 또는 20
해석: "추천한 K개 중 사용자가 실제로 좋아한 비율"
```

**NDCG@K** (Normalized Discounted Cumulative Gain)
```
DCG@K = Σ (2^rel_i - 1) / log₂(i+1)
NDCG@K = DCG@K / IDCG@K

해석: "상위 추천일수록 더 중요하게 평가"
```

**Hit Rate@K**
```
HR@K = |Users with at least 1 hit| / |Total users|

해석: "최소 1개라도 맞춘 사용자 비율"
```

**Precision@K**
```
Precision@K = |Recommended ∩ Relevant| / K

주의: Recall보다 덜 중요 (추천 시스템에선)
```

**Coverage**
```
Coverage = |Unique items recommended| / |Total items|

해석: "얼마나 다양한 영화를 추천했는가"
목표: > 30% (너무 낮으면 인기 영화만 추천)
```

### 7.2 평가 프로토콜

**Leave-One-Out Strategy:**
```python
for user in test_users:
    # 1. 사용자가 안 본 영화 99개 랜덤 샘플
    # 2. Test set의 positive item 1개 추가 (총 100개)
    # 3. 100개 중 Top-K 추천
    # 4. Positive item이 Top-K에 있는지 확인
```

**Full Ranking Strategy:** (더 엄격)
```python
for user in test_users:
    # 1. 사용자가 안 본 모든 영화 (~10,000개)
    # 2. 모두 점수 매기기
    # 3. Top-K 추천
    # 4. Test set positive items와 비교
```

**추천:** 계산 비용을 고려해 Leave-One-Out으로 시작

### 7.3 Baseline 비교

**필수 Baseline:**
1. **Random:** 랜덤 추천
2. **Popularity:** 인기 영화 순위대로 추천
3. **Matrix Factorization (SVD):** 전통적 방법
4. **User-KNN:** User-based collaborative filtering

**목표:** GNN이 모든 baseline을 이겨야 함

---

## 8. Breakthrough 전략

### 8.1 Negative Sampling 개선

**전략 1: Hard Negative Mining**
```python
# Epoch마다 현재 모델로 점수 높지만 실제론 negative인 샘플 선택
hard_negatives = items_with_high_scores_but_not_interacted

효과: 모델이 어려운 케이스에 집중 학습
```

**전략 2: Popularity-based Sampling**
```python
# 인기 영화 중에서 negative sampling (더 어려운 negative)
prob = popularity^α  # α=0.75 (popular items 더 자주 샘플)

효과: 인기 편향 줄이고 discrimination 향상
```

### 8.2 Graph Augmentation

**Item-Item Graph 추가:**
```python
# Co-occurrence graph
if (user watched item_i AND item_j):
    add_edge(item_i, item_j, weight=frequency)

효과: 영화 간 유사도 정보 활용
```

**구현:**
```python
# Heterogeneous graph
edges = {
    ('user', 'watches', 'item'): user_item_edges,
    ('item', 'similar', 'item'): item_item_edges
}
```

### 8.3 Ensemble 전략

**Model Ensemble:**
```python
# 다른 random seed로 5개 모델 학습
models = [model_seed1, model_seed2, ..., model_seed5]

# Prediction averaging
final_score = mean([m.predict(u, i) for m in models])

효과: Variance 감소, 안정적 성능
```

**Method Ensemble:**
```python
# LightGCN + Matrix Factorization
score_final = 0.7 * score_gnn + 0.3 * score_mf

효과: 서로 다른 관점의 장점 결합
```

### 8.4 고급 기법

**1. Contrastive Learning**
- 같은 사용자의 다른 augmented view를 가깝게 학습
- Self-supervised learning으로 representation 향상

**2. Knowledge Distillation**
- 큰 teacher model → 작은 student model
- 추론 속도 향상

**3. Meta-Learning**
- Cold-start 문제 해결
- Few-shot learning으로 새 아이템 빠르게 학습

**우선순위:** 1 > 2 > 3 (성능 vs 복잡도)

---

## 9. 실험 계획 (4주)

### Week 1: 기반 구축
```
Day 1-2: 데이터 전처리
- EDA 심화 분석
- Train/Val/Test split
- Negative sampling dataset 생성

Day 3-4: Baseline 구현
- Random, Popularity baseline
- Matrix Factorization (SVD)
- 평가 지표 구현

Day 5-7: 평가 프레임워크
- Recall@K, NDCG@K 구현
- Visualization (학습 곡선, 분포 등)
```

### Week 2: GNN 구현
```
Day 1-3: LightGCN 구현
- PyTorch Geometric 활용
- Graph 구축
- Forward/Backward pass 검증

Day 4-5: BPR Loss 학습
- Training loop
- Validation monitoring

Day 6-7: 첫 번째 실험
- Baseline과 비교
- 문제점 파악
```

### Week 3: 최적화
```
Day 1-3: Hyperparameter Tuning
- Grid search or Random search
- Embedding dim, layers, lr 등

Day 4-5: Ablation Study
- Layer 수 영향
- Negative sampling ratio 영향
- Rating 사용 유무

Day 6-7: Advanced Techniques
- Hard negative sampling 시도
- Graph augmentation 실험
```

### Week 4: 완성
```
Day 1-2: Ensemble 구현
- Multiple models 학습
- Ensemble 전략 선정

Day 3-4: 최종 모델 선정
- Validation 결과 종합
- Best configuration 확정

Day 5-6: Test 및 제출 준비
- Test data 예측
- 출력 형식 구현 (user, item, O/X)
- 모델 저장 (.pt)

Day 7: 문서화 및 검증
- 코드 정리
- README 작성
- 최종 점검
```

---

## 10. 구현 체크리스트

### 데이터 처리
- [ ] Train/Val/Test split (80/10/10)
- [ ] User/Item ID re-indexing
- [ ] Negative sampling 구현
- [ ] Graph edge list 생성
- [ ] Data loader 구현

### 모델
- [ ] LightGCN 구현
- [ ] Embedding layer 초기화
- [ ] Graph convolution layer
- [ ] Prediction layer
- [ ] L2 regularization

### 학습
- [ ] BPR Loss 구현
- [ ] Training loop
- [ ] Validation loop
- [ ] Checkpoint 저장
- [ ] Early stopping

### 평가
- [ ] Recall@K 구현
- [ ] NDCG@K 구현
- [ ] Hit Rate@K 구현
- [ ] Coverage 계산
- [ ] Baseline 비교

### 최종 제출
- [ ] Test data 로드
- [ ] 예측 수행
- [ ] O/X 형식 출력
- [ ] 모델 저장 (.pt)
- [ ] 통계 출력 (Total recommends)

---

## 11. 예상 도전과제 및 해결책

### 도전과제 1: 극심한 희소성
**문제:** 1.5% density → 대부분의 user-item 조합 미관측
**해결책:**
- Negative sampling ratio 높이기 (1:4)
- Graph convolution으로 indirect connection 활용
- Regularization 강화 (dropout, weight decay)

### 도전과제 2: 롱테일 분포
**문제:** 대부분 영화가 3회 이하만 시청
**해결책:**
- 최소 상호작용 threshold 설정 (5회 미만 영화 제거 고려)
- Popularity debiasing
- Item-item graph로 정보 보강

### 도전과제 3: Overfitting
**문제:** 파워유저 몇 명이 데이터 지배
**해결책:**
- 파워유저 데이터 샘플링
- Strong regularization (L2, dropout)
- Early stopping (patience=20)

### 도전과제 4: Cold-start
**문제:** 새로운 사용자/영화 추천 어려움
**해결책:**
- Popularity-based fallback
- Graph augmentation으로 indirect information
- Meta-learning (고급)

---

## 12. 성공 기준

### 최소 목표 (필수)
- [x] Recall@10 ≥ 0.20
- [x] NDCG@10 ≥ 0.25
- [x] Baseline (MF) 대비 10% 이상 성능 향상
- [x] Coverage ≥ 20%

### 목표 (기대)
- [x] Recall@10 ≥ 0.30
- [x] NDCG@10 ≥ 0.35
- [x] Baseline 대비 20% 이상 성능 향상
- [x] Coverage ≥ 30%

### 우수 목표 (도전)
- [x] Recall@10 ≥ 0.40
- [x] NDCG@10 ≥ 0.45
- [x] Ensemble로 추가 5% 향상
- [x] Coverage ≥ 40%

---

## 13. 참고 자료

### 핵심 논문
1. **LightGCN** (SIGIR 2020): "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
2. **BPR** (UAI 2009): "BPR: Bayesian Personalized Ranking from Implicit Feedback"
3. **NGCF** (SIGIR 2019): "Neural Graph Collaborative Filtering"

### 구현 참고
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- RecBole (추천 시스템 라이브러리): https://recbole.io/

### 데이터셋 벤치마크
- MovieLens (유사 데이터): 일반적으로 Recall@20 = 0.3~0.5
- Gowalla (체크인 데이터): Recall@20 = 0.15~0.25

**현재 데이터 예상 성능:**
- Baseline (MF): Recall@10 ≈ 0.18
- LightGCN: Recall@10 ≈ 0.25~0.35
- LightGCN + Ensemble: Recall@10 ≈ 0.30~0.40

---

## 14. 결론 및 핵심 원칙

### 핵심 원칙
1. **단순함이 강력하다**: LightGCN이 복잡한 모델보다 자주 이김
2. **데이터가 왕이다**: Negative sampling 전략이 성능 좌우
3. **검증이 중요하다**: Validation으로 overfitting 방지
4. **비교가 필수다**: Baseline 없이는 성공 판단 불가

### 최종 추천 Configuration
```python
model = LightGCN(
    num_users=668,
    num_items=10321,
    embedding_dim=64,
    num_layers=3,
    dropout=0.1
)

optimizer = Adam(lr=1e-3, weight_decay=1e-4)
loss_fn = BPR_Loss()
negative_ratio = 4
batch_size = 1024
epochs = 200 (with early stopping)
```

### 성공 공식
```
Good Data Preprocessing
+ Simple but Effective Model (LightGCN)
+ Smart Negative Sampling
+ Careful Hyperparameter Tuning
+ Ensemble (if needed)
= Top Performance
```

**행운을 빕니다! 🚀**