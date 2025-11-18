# GNN 추천 시스템 브레인스토밍 v1
**작성일**: 2025-11-18
**목표**: A ver (Binary Classification), B ver (Rating Prediction) 두 가지 접근 방식으로 추천 시스템 구현

---

## 1. 데이터 분석 및 특성

### 1.1 기본 통계
- **Users**: 668명
- **Items**: 10,321개
- **Interactions**: 105,139개
- **Sparsity**: ~98.5% (매우 희소)
- **User당 평균 interactions**: 157.4개
- **Item당 평균 interactions**: 10.2개
- **Cold users (≤10 interactions)**: 0명 (모든 유저가 충분한 데이터 보유)

### 1.2 Rating 분포
```
Rating 0.5: 1,189 (1.1%)
Rating 1.0: 3,254 (3.1%)
Rating 1.5: 1,564 (1.5%)
Rating 2.0: 7,929 (7.5%)
Rating 2.5: 5,473 (5.2%)
Rating 3.0: 21,676 (20.6%)
Rating 3.5: 12,224 (11.6%)
Rating 4.0: 28,831 (27.4%) ← Threshold 기준
Rating 4.5: 8,174 (7.8%)
Rating 5.0: 14,825 (14.1%)
```

**핵심 인사이트**:
- Rating >= 4: 51,830개 (49.3%) - **거의 50:50 비율**
- Rating < 4: 53,309개 (50.7%)
- Rating 3.0이 가장 많음 (20.6%)
- 평가가 양극화되지 않고 전체적으로 분산됨

### 1.3 데이터 특성 요약
1. **충분한 데이터**: Cold start 문제 없음 (모든 유저가 10개 이상 interaction)
2. **높은 Sparsity**: 98.5% → GNN의 message passing이 중요
3. **Balanced Rating**: Rating 4 기준으로 positive/negative가 거의 50:50
4. **영화 추천 도메인**: Sequential pattern보다는 user-item affinity가 중요

---

## 2. A ver vs B ver 전략 비교

### 2.1 A ver: Binary Classification (연결 존재 여부)

#### 개념
- **Ground Truth**: User-item edge 존재 = Positive (rating 무시)
- **Task**: 존재하지 않는 user-item 쌍에 대해 "연결될 가능성" 예측
- **출력**: O (추천) / X (비추천)

#### 장점
- 단순하고 직관적
- 모든 interaction을 positive로 활용 (데이터 최대 활용)
- 협업 필터링의 고전적 접근

#### 단점
- **낮은 rating도 positive로 간주** → 품질 낮은 추천 가능
- 예: Rating 0.5도 "추천"이 됨 (실제로는 싫어한 영화)
- 새로운 평가 규칙에서 불리: "잘못된 추천은 감점"

#### 적용 방법
- **Positive edges**: train.csv의 모든 user-item 쌍
- **Negative sampling**: 연결되지 않은 쌍 중 랜덤 샘플링
- **Loss**: BPR Loss (Bayesian Personalized Ranking)
- **Evaluation**: Precision, Recall, F1 (이진 분류 metrics)

---

### 2.2 B ver: Rating Prediction + Threshold (품질 고려)

#### 개념
- **Ground Truth**: Rating >= 4 = Positive (좋은 영화만)
- **Task**: User-item 쌍의 예상 rating 예측
- **출력**: 예측 rating >= 4 → O, < 4 → X

#### 장점
- **품질 있는 추천**: 좋은 영화만 추천
- 새로운 평가 규칙에 유리: "잘못된 추천 감점" 최소화
- Rating 정보를 명시적으로 활용
- 기존 V9b와 유사한 접근 (proven performance)

#### 단점
- Positive 데이터가 절반으로 줄어듦 (51,830개만 사용)
- Rating < 4인 데이터의 활용 방법 고민 필요

#### 적용 방법
- **Positive edges**: Rating >= 4인 쌍만
- **Negative edges**: Rating < 4인 쌍 + 연결 없는 쌍
- **Loss**:
  - Option 1: BPR Loss (positive vs negative)
  - Option 2: MSE Loss (rating 값 직접 예측)
  - Option 3: Hybrid Loss (BPR + MSE)
- **Evaluation**: AUC-ROC, Precision, Recall, F1 (rating >= 4 기준)

---

## 3. 데이터 전처리 및 가공 전략

### 3.1 ID 매핑 및 정규화
```python
# User/Item ID를 0-based index로 변환
user2idx = {u: i for i, u in enumerate(sorted(df['user'].unique()))}
item2idx = {it: i for i, it in enumerate(sorted(df['item'].unique()))}
```

### 3.2 Train/Val/Test Split 전략

#### A ver (모든 edge 사용)
```
- 전체 interactions을 7:1.5:1.5 비율로 split
- Train: 73,597개
- Val: 15,771개
- Test: 15,771개
```

#### B ver (Rating >= 4만 split, Rating < 4는 train graph에만 포함)
```
- Rating >= 4만 split (V9b 방식)
- Rating < 4는 모두 train graph에 포함 (학습용)
- Train: 36,281 (positive) + 53,309 (negative for graph) = 89,590개
- Val: 7,775개 (rating >= 4만)
- Test: 7,774개 (rating >= 4만)
```

**핵심 아이디어**: Rating < 4도 그래프 구조 학습에는 유용함 (단, 평가 시엔 제외)

### 3.3 Negative Sampling 전략

#### Random Negative Sampling (Baseline)
- 연결되지 않은 user-item 쌍을 랜덤 샘플링
- 빠르고 간단
- 단점: 쉬운 negative만 학습 (informative하지 않음)

#### Hard Negative Mining (Advanced)
- **현재 모델이 헷갈리는** negative 샘플링
- 높은 score를 받았지만 실제로는 negative인 쌍
- V9b에서 효과 입증 (+0.7% AUC-ROC)
- 구현:
  ```python
  # 1. 많은 candidate 생성 (10배)
  # 2. 모델로 score 계산
  # 3. 상위 K개를 hard negative로 선택
  ```

#### Hybrid Sampling (Best Practice)
- 50% Hard + 50% Random
- Hard만 사용 시 overfitting 위험
- Random 섞어서 다양성 확보
- **우리 전략**: 이 방식 채택

### 3.4 Graph 구성 전략

#### Option 1: Unweighted Graph (A ver 기본)
- 모든 edge weight = 1
- 단순하고 빠름

#### Option 2: Rating Weighted Graph (B ver 추천)
- Edge weight = f(rating)
- V9b 방식: `weight = 0.4 + 0.15 * rating`
  - Rating 1 → weight 0.55
  - Rating 3 → weight 0.85
  - Rating 5 → weight 1.15
- 높은 rating에 더 강한 신호 부여
- **효과 입증**: V9b에서 성능 향상

#### Option 3: Attention-based Weighting
- 학습 가능한 attention으로 weight 결정
- 복잡하지만 더 flexible
- 구현 난이도 높음

**우리 선택**:
- **A ver**: Unweighted (단순성)
- **B ver**: Rating Weighted (proven method)

---

## 4. 임베딩 및 GNN 아키텍처 전략

### 4.1 임베딩 차원 선택

#### 기존 V9b: EMB_DIM = 32
- 파라미터: 351,648개
- Overfitting 방지
- 학습 속도 빠름
- 성능 우수 (AUC-ROC 0.9264)

#### 저차원 vs 고차원 논쟁
**High-Dimensional Data Analysis with Low-Dimensional Models 관점**:
> "현실의 많은 고차원 문제는 사실 저차원 구조를 가지고 있다"

**우리 데이터 분석**:
- User: 668개 → 저차원으로 충분
- Item: 10,321개 → 상대적으로 많지만, collaborative filtering의 본질은 user-item affinity (저차원 latent space에 존재)
- 기존 32차원으로도 0.926 AUC 달성 → **저차원 가설 지지**

**실험 계획**:
- **cca1/ccb1**: EMB_DIM = 32 (V9b 계승, baseline)
- **향후 실험**: 16, 64 비교

**예상**: 32가 sweet spot일 가능성 높음 (V9b 검증)

### 4.2 GNN 모델 선택

#### LightGCN (현재 사용 중)
- **장점**:
  - 단순하고 효과적
  - Self-connection 없음 (collaborative signal만 사용)
  - Layer normalization으로 안정적
  - 추천 시스템에서 SOTA 수준
  - V9b에서 검증됨 (0.9264 AUC)

- **수식**:
  ```
  e^(k+1) = Σ_{j∈N(i)} (1/√|N(i)|√|N(j)|) * e^(k)_j
  final_e = mean([e^(0), e^(1), ..., e^(K)])
  ```

- **핵심**: Weighted sum of multi-hop neighbors

#### 다른 GNN 대안들

**1. GCN (Graph Convolutional Network)**
- 더 일반적인 그래프 학습
- 단점: Self-loop + non-linear activation → 추천에서 LightGCN보다 낮은 성능
- **채택 안함**: LightGCN이 추천에 최적화됨

**2. GraphSAGE**
- Neighbor sampling으로 scalability 확보
- 단점: 우리 데이터는 668 users로 작음 (scalability 불필요)
- **채택 안함**: Overkill

**3. GAT (Graph Attention Network)**
- Attention으로 neighbor importance 학습
- 장점: More expressive
- 단점: 복잡하고 느림, overfitting 위험
- **보류**: LightGCN으로 먼저 시도 후 고려

**우리 선택**: **LightGCN** (proven, simple, effective)

### 4.3 Layer 수 선택

#### V9b: N_LAYERS = 2
- 2-hop neighbors까지 정보 수집
- 충분한 collaborative signal
- 학습 안정적

#### Layer 수에 따른 trade-off
- **1-layer**: 직접 연결만 (under-smoothing)
- **2-layer**: 2-hop (sweet spot for most datasets)
- **3-layer**: 3-hop (over-smoothing 위험)
- **4+ layers**: 거의 모든 노드가 비슷해짐 (over-smoothing)

**우리 선택**: **N_LAYERS = 2** (V9b 검증)

### 4.4 Normalization 전략

#### Symmetric Normalization (LightGCN 기본)
```python
# Degree normalization
deg_inv_sqrt = deg^(-0.5)
weight = deg_inv_sqrt[i] * deg_inv_sqrt[j]
```

- Popular item/user가 너무 dominant하지 않도록
- 균형잡힌 message passing

**우리 선택**: LightGCN 기본 방식 유지

---

## 5. Loss Function 전략

### 5.1 A ver (Binary Classification)

#### BPR Loss (Bayesian Personalized Ranking)
```python
loss = -log(sigmoid(score_pos - score_neg))
```

**장점**:
- Pairwise ranking loss
- Positive가 negative보다 높은 score를 가지도록 학습
- 추천 시스템에서 표준

**우리 선택**: **BPR Loss** (A ver)

### 5.2 B ver (Rating Prediction)

#### Option 1: BPR Loss (Ranking-based)
- Rating >= 4를 positive로, < 4를 negative로
- 단순하고 효과적
- V9b에서 검증됨

#### Option 2: MSE Loss (Regression)
```python
loss = MSE(predicted_rating, actual_rating)
```
- Rating 값 자체를 예측
- 장점: Rating 정보를 직접 활용
- 단점: Regression task로 전환 (추천은 ranking task)

#### Option 3: Hybrid Loss
```python
loss = BPR_loss + λ * MSE_loss
```
- Ranking과 rating prediction 동시 학습
- λ = 0.1 ~ 0.5 정도
- 복잡하지만 더 정확할 수 있음

**우리 선택**:
- **ccb1**: BPR Loss만 (baseline, V9b 방식)
- **향후**: Hybrid Loss 실험

---

## 6. Evaluation Metrics 전략

### 6.1 A ver Metrics

#### Binary Classification Metrics
1. **Precision**: 추천한 것 중 실제 positive 비율
   - 중요도: ★★★ (잘못된 추천 감점 규칙)
2. **Recall**: 실제 positive 중 추천한 비율
   - 중요도: ★★
3. **F1 Score**: Precision과 Recall의 조화평균
   - 중요도: ★★★ (균형잡힌 평가)
4. **AUC-ROC**: Positive/Negative 구분 능력
   - 중요도: ★★★ (threshold-independent)

#### Ranking Metrics
5. **Precision@K**: Top-K 추천의 정확도
6. **Recall@K**: Top-K로 커버한 positive 비율
7. **Hit@K**: Top-K에 최소 1개라도 positive가 있는지

**우리 선택**: AUC-ROC + F1 + Precision@K (종합 평가)

### 6.2 B ver Metrics

#### Classification Metrics (Rating >= 4 기준)
1. **AUC-ROC**: 좋은 영화 vs 나쁜 영화 구분 능력
   - 중요도: ★★★★ (V9b primary metric)
2. **Precision**: 추천한 것 중 실제 rating >= 4 비율
   - 중요도: ★★★
3. **Recall**: Rating >= 4 중 추천한 비율
   - 중요도: ★★
4. **F1 Score**: 균형잡힌 평가
   - 중요도: ★★★
5. **Accuracy**: 전체 정확도
   - 중요도: ★

#### Regression Metrics (Rating 값 예측 시)
6. **RMSE**: Root Mean Squared Error
7. **MAE**: Mean Absolute Error

**우리 선택**: AUC-ROC (primary) + F1 + Precision

---

## 7. Breakthrough Ideas

### 7.1 저차원 모델링 (Low-Dimensional Models)

#### 핵심 아이디어
> "10,321개 item이지만, 실제 user preference는 몇 개의 latent factor로 설명 가능"

**예시**:
- 장르 선호도: 액션, 로맨스, 코미디, ... (5-10개 factor)
- 시대 선호도: 구작 vs 신작 (1-2개 factor)
- 분위기: 밝음 vs 어두움 (1-2개 factor)

**실제 차원**: ~10-20개 정도의 latent factor로 대부분 설명 가능

**V9b 검증**: 32차원으로 0.9264 AUC → 저차원으로 충분

**우리 전략**:
- 불필요하게 큰 embedding 사용 안함
- 32차원 baseline, 향후 16차원도 실험
- Regularization으로 실제 사용 차원 더욱 줄이기 (L2, Dropout)

### 7.2 Graph Structure Exploitation

#### User-User / Item-Item Similarity
- 현재: Bipartite graph (user-item만)
- 개선 아이디어: User-user, Item-item edge 추가
  - User similarity: Jaccard, Cosine
  - Item co-occurrence

**문제**: 추가 edge로 graph 복잡해짐 (학습 느려짐)

**대안**: Implicit similarity (LightGCN의 multi-hop이 이미 수행)

**우리 선택**: 일단 bipartite graph만 (단순성)

### 7.3 Rating Distribution Modeling

#### 관찰
- Rating 분포가 전체적으로 퍼져있음 (0.5~5.0)
- 각 rating value가 다른 의미를 가짐

#### 아이디어
- Rating을 categorical로 취급
- 10개 class classification (0.5, 1.0, ..., 5.0)
- Cross-entropy loss

**장점**: Rating의 ordinal nature 활용

**단점**: 복잡함, BPR보다 나을지 불확실

**우리 선택**: 보류 (향후 실험)

### 7.4 Cold Start Handling (필요 없음)

- 모든 user가 10개 이상 interaction
- Cold start 문제 없음
- **Skip**

### 7.5 Temporal Dynamics (순서 정보)

#### 관찰
- Train.csv에 timestamp 없음
- 영화 추천은 sequential pattern이 덜 중요 (음악/뉴스와 다름)

**우리 선택**: Temporal 고려 안함 (데이터 없음 + 필요성 낮음)

### 7.6 Multi-Task Learning

#### 아이디어
- Task 1: Binary classification (연결 여부)
- Task 2: Rating prediction (값 예측)
- 동시 학습으로 representation 향상

**구현**:
```python
loss = loss_binary + λ * loss_rating
```

**장점**: 더 rich한 representation

**단점**: 복잡함, hyperparameter tuning 필요

**우리 선택**: 보류 (향후 실험)

### 7.7 Ensemble (A ver + B ver)

#### 최종 아이디어
- A ver와 B ver를 독립적으로 학습
- 예측 시 ensemble:
  - Averaging
  - Voting
  - Stacking

**기대 효과**:
- A ver: Recall 높음 (모든 interaction 활용)
- B ver: Precision 높음 (품질 고려)
- Ensemble: Best of both worlds

**우리 선택**: 일단 각각 학습 후 성능 비교, 나중에 ensemble 고려

---

## 8. 추천 출력 전략 (새로운 평가 규칙 대응)

### 8.1 평가 규칙 요약
```
- 기존 선택 item 수의 20% 이하 개수까지 추천
- User별로 추천 개수가 다름
- Item과 interaction이 10개 이하인 user: 무조건 2개 추천
- 잘못된 추천은 감점
```

### 8.2 Hybrid 추천 전략 (V9b 방식)

#### Step 1: Threshold Filtering
- Score > θ인 item만 후보로 선택
- θ는 validation set에서 F1 최대화로 결정

#### Step 2: Top-K Selection
- Threshold 통과한 후보 중 Top-K만 추천
- K = max(2, interaction_count * 0.2)
- K는 user별로 다름

#### Step 3: 출력
- 선택된 K개: O
- 나머지: X

**장점**:
- Precision 높임 (threshold로 필터링)
- User별 제약 만족 (K 개수 조정)
- Flexible (0~K개 추천 가능)

**V9b 결과**:
- Threshold = 0.4872
- O ratio: 50.9%
- Precision: 0.8646
- Recall: 0.8794

**우리 선택**: 이 방식 채택 (검증됨)

### 8.3 Conservative 전략 (Precision 우선)

#### 아이디어
- Threshold를 더 높게 설정
- 확실한 것만 추천
- Precision 극대화, Recall 희생

**적용 상황**:
- "잘못된 추천은 감점"이 강하게 적용될 때
- Test set이 까다로울 것으로 예상될 때

**구현**:
- Threshold 1.2배 증가
- 또는 F1 대신 Precision 최대화로 튜닝

### 8.4 Aggressive 전략 (Recall 우선)

#### 아이디어
- Threshold를 낮게 설정
- 더 많이 추천
- Recall 극대화

**적용 상황**:
- "추천 많이 해도 됨" (Q&A 참고)
- Coverage 중요할 때

**우리 선택**:
- Baseline: F1 최대화 (균형)
- 상황에 따라 조정 가능하도록 구현

---

## 9. 구현 계획

### 9.1 cca1.ipynb (A ver: Binary Classification)

#### 목표
- 모든 interaction을 positive로 활용
- Binary classification task
- Baseline 성능 확인

#### 구현 내용
1. **데이터 전처리**
   - 모든 user-item 쌍을 positive로
   - Train/Val/Test split (7:1.5:1.5)

2. **Graph 구성**
   - Unweighted bipartite graph
   - Symmetric normalization

3. **모델**
   - LightGCN (EMB_DIM=32, N_LAYERS=2)

4. **학습**
   - Loss: BPR Loss
   - Negative sampling: 50% Hard + 50% Random
   - Optimizer: Adam (LR=5e-3)
   - Epochs: 50

5. **평가**
   - AUC-ROC
   - F1 Score
   - Precision/Recall
   - Precision@K, Recall@K, Hit@K

6. **추천 출력**
   - Hybrid (Threshold + Top-K)
   - F1 최대화 threshold
   - sample1.csv, sample2.csv 테스트

#### 예상 성능
- AUC-ROC: 0.88 ~ 0.92
- F1: 0.80 ~ 0.85
- V9b보다 약간 낮을 것 (품질 고려 안함)

### 9.2 ccb1.ipynb (B ver: Rating Prediction)

#### 목표
- Rating >= 4만 positive로
- 품질 있는 추천
- V9b 성능 재현 및 개선

#### 구현 내용
1. **데이터 전처리**
   - Rating >= 4: Positive
   - Rating < 4: Negative (train graph에 포함, 평가 제외)
   - Train/Val/Test split (rating >= 4만)

2. **Graph 구성**
   - Rating Weighted Graph
   - Weight = 0.4 + 0.15 * rating

3. **모델**
   - LightGCN (EMB_DIM=32, N_LAYERS=2)

4. **학습**
   - Loss: BPR Loss (rating >= 4 vs < 4)
   - Negative sampling: 50% Hard + 50% Random
   - Optimizer: Adam (LR=5e-3)
   - Epochs: 50

5. **평가**
   - AUC-ROC (primary)
   - F1 Score
   - Precision/Recall
   - Accuracy

6. **추천 출력**
   - Hybrid (Threshold + Top-K)
   - F1 최대화 threshold
   - sample1.csv 테스트

#### 예상 성능
- AUC-ROC: 0.92 ~ 0.93 (V9b 수준)
- F1: 0.86 ~ 0.88
- Precision 높음 (품질 고려)

### 9.3 코드 구조 및 문서화

#### 파일 구조
```
cc/
├── cca1.ipynb          # A ver: Binary Classification
├── ccb1.ipynb          # B ver: Rating Prediction
└── common.py           # 공통 유틸리티 (향후 필요 시)

cc_docs/
├── brainstorming_v1.md     # 이 문서
├── cca1_results.md         # A ver 결과 및 분석 (작성 예정)
├── ccb1_results.md         # B ver 결과 및 분석 (작성 예정)
└── comparison.md           # A vs B 비교 (작성 예정)

cc_models/
├── cca1_best.pt            # A ver 최고 모델
└── ccb1_best.pt            # B ver 최고 모델
```

#### 노트북 구조 (공통)
1. **Introduction**: 목표, 전략 요약
2. **Data Preprocessing**: 데이터 로드, split, mapping
3. **Graph Construction**: Edge index, weights
4. **Model Definition**: LightGCN, Loss function
5. **Training**: Loop, evaluation
6. **Validation**: Threshold tuning, metrics
7. **Test Evaluation**: Final performance
8. **Sample Prediction**: sample1.csv, sample2.csv 출력
9. **Summary**: 결과 요약, 개선 방향

#### 문서화 원칙
- 각 cell에 markdown 설명 추가
- Hyperparameter 명시
- 결과 시각화 (loss curve, metrics)
- 중요한 인사이트는 문서로 정리

---

## 10. 향후 실험 방향

### 10.1 v2 (개선 방향)

#### cca2 / ccb2 아이디어
1. **Embedding dimension 실험**: 16, 64 비교
2. **Layer 수 실험**: 1, 3 비교
3. **Hybrid Loss**: BPR + MSE (B ver)
4. **Learning rate schedule**: Cosine annealing
5. **Ensemble**: cca1 + ccb1

### 10.2 Advanced Features

1. **Item/User Features**
   - Item metadata (genre, year, ...) 없음
   - User demographics 없음
   - → Feature engineering 불가
   - → Pure collaborative filtering

2. **Graph Attention**
   - GAT 시도
   - Attention weight 학습
   - 복잡도 증가 주의

3. **Multi-Task Learning**
   - Binary + Rating 동시 학습
   - Shared embedding

### 10.3 Hyperparameter Tuning

- Learning rate: [1e-3, 5e-3, 1e-2]
- Weight decay: [0, 1e-6, 1e-5, 1e-4]
- Negative sampling ratio: [0.3, 0.5, 0.7]
- Batch size: [512, 1024, 2048]

---

## 11. 리스크 및 대응

### 11.1 Overfitting
**증상**: Train loss 낮지만 val 성능 정체
**대응**:
- Weight decay 증가
- Early stopping
- Embedding dimension 감소
- Dropout 추가 (LightGCN에는 기본적으로 없음)

### 11.2 Underfitting
**증상**: Train/Val 모두 성능 낮음
**대응**:
- Learning rate 증가
- Epochs 증가
- Model capacity 증가 (embedding dim, layers)

### 11.3 Hard Negative Mining 실패
**증상**: Loss 증가, 성능 하락
**대응**:
- Hard ratio 감소 (50% → 30%)
- Random negative로 대체
- Hard negative candidate 수 조정

### 11.4 Threshold 선택 실패
**증상**: Precision/Recall 극단적 불균형
**대응**:
- F1 대신 F2 (recall 중시) 또는 F0.5 (precision 중시)
- Validation set 크기 증가
- Multiple threshold 시도

---

## 12. 성공 기준

### 12.1 Baseline 목표 (cca1 / ccb1)

#### A ver (cca1)
- **AUC-ROC >= 0.88**
- **F1 >= 0.80**
- Precision >= 0.75
- Sample test 통과 (5/5 예측)

#### B ver (ccb1)
- **AUC-ROC >= 0.92** (V9b 수준)
- **F1 >= 0.86**
- Precision >= 0.85
- Sample test 통과 (정답: XXOXO for sample1.csv)

### 12.2 비교 분석
- A ver vs B ver 성능 비교
- Trade-off 분석 (Precision vs Recall)
- 어느 ver가 새로운 평가 규칙에 유리한지 판단

### 12.3 최종 목표
- 두 버전 모두 구현 및 평가 완료
- 문서화 완료
- 교수님 test file에 대비한 inference 코드 준비
- (Optional) Ensemble 또는 최고 성능 모델 선택

---

## 13. 참고 자료

### 13.1 논문
- LightGCN: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)
- BPR: "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
- Hard Negative Mining: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019)

### 13.2 기존 작업
- V9b 노트북: claude_made/recsys_v9b.ipynb
- V9a 노트북: claude_made/recsys_v9a.ipynb
- V8b 노트북: claude_made/recsys_v8b.ipynb
- plan.md: 기존 개선 계획

### 13.3 책
- "High-Dimensional Data Analysis with Low-Dimensional Models" (book.md)
  - 저차원 모델링의 중요성
  - 우리 프로젝트와의 연관성: Embedding dimension 선택

---

## 14. 타임라인 (예상)

1. **Day 1**: cca1.ipynb 구현 및 학습 (2-3시간)
2. **Day 1**: ccb1.ipynb 구현 및 학습 (2-3시간)
3. **Day 2**: 결과 분석 및 문서화 (1-2시간)
4. **Day 2**: A vs B 비교 및 개선 방향 도출 (1시간)
5. **Day 3+**: v2 실험 (선택적)

---

## 15. 결론

### 15.1 핵심 전략
1. **두 가지 접근**: A ver (모든 edge) vs B ver (rating >= 4만)
2. **검증된 모델**: LightGCN (V9b 검증)
3. **저차원 임베딩**: 32차원 (overfitting 방지)
4. **Hard Negative Mining**: 50% hard + 50% random
5. **Hybrid 추천**: Threshold + Top-K (평가 규칙 대응)

### 15.2 기대 효과
- **A ver**: 높은 recall (모든 데이터 활용)
- **B ver**: 높은 precision (품질 고려)
- **비교 분석**: 어느 접근이 더 효과적인지 판단
- **Ensemble 가능성**: 두 ver 결합으로 best performance

### 15.3 Next Steps
1. ✅ 브레인스토밍 완료
2. ⏳ cca1.ipynb 구현
3. ⏳ ccb1.ipynb 구현
4. ⏳ 결과 분석 및 문서화

---

**문서 버전**: v1.0
**최종 수정**: 2025-11-18
**작성자**: Claude (AI Assistant)
