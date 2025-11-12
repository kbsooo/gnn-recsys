# 📊 Exploratory Data Analysis Report
## GNN-based Recommendation System

---

## 1. 데이터셋 개요

### 기본 정보
- **총 상호작용 수**: 105,139개
- **고유 사용자 수**: 668명
- **고유 영화 수**: 10,321개
- **데이터 형식**: User-Item-Rating 삼중항 (Triple)

### 데이터 밀도
- **Density**: 1.525%
- **Sparsity**: 98.475%
- **의미**: 매우 희소한(sparse) 데이터셋 → GNN이 효과적으로 작동할 수 있는 환경

---

## 2. 사용자(User) 분석

### 상호작용 통계
| 지표 | 값 |
|------|-----|
| 최소 상호작용 | 20개 |
| 최대 상호작용 | 5,672개 |
| 평균 | 157.4개 |
| 중앙값 | 70개 |
| 표준편차 | 319.3 |

### 주요 발견사항

**1) 사용자 활동도의 높은 분산**
- 가장 활발한 사용자(User 668): 5,672개 상호작용
- 중앙값(70개)과 평균(157개)의 차이 → **Right-skewed distribution**
- 소수의 매우 활발한 사용자가 존재

**2) Percentile 분석**
```
10%: 25개
25%: 35개
50%: 70개
75%: 153개
90%: 334개
95%: 550개
99%: 1,197개
```
- 상위 10% 사용자가 매우 활발함
- 하위 50%는 상대적으로 적은 상호작용

**3) 사용자 평점 경향**
- 평균 평점 범위: 1.98 ~ 5.00
- 전체 평균: 3.66
- 중앙값: 3.69
- 표준편차: 0.46

**사용자 유형 분류:**
- **High raters (평균 ≥4.0)**: 164명 (24.6%)
- **Medium raters (3.0-4.0)**: 451명 (67.5%)
- **Low raters (<3.0)**: 53명 (7.9%)

→ 대부분의 사용자가 중간~긍정적 평가 경향

---

## 3. 아이템(영화) 분석

### 인기도 통계
| 지표 | 값 |
|------|-----|
| 최소 상호작용 | 1개 |
| 최대 상호작용 | 324개 |
| 평균 | 10.2개 |
| 중앙값 | 3개 |
| 표준편차 | 22.8 |

### Long-tail 분포 특성

**심각한 Long-tail 문제:**
- **1회만 상호작용**: 3,640개 아이템 (35.3%)
- **≤5회 상호작용**: 6,952개 아이템 (67.4%)
- **≤10회 상호작용**: 8,162개 아이템 (79.1%)

→ **약 80%의 영화가 10번 이하의 상호작용만 가짐**

### Top 20 인기 영화

| 순위 | Item ID | 상호작용 | 평균 평점 |
|------|---------|----------|-----------|
| 1 | 296 | 324 | 4.16 |
| 2 | 356 | 311 | 4.14 |
| 3 | 318 | 307 | 4.45 |
| 4 | 480 | 294 | 3.66 |
| 5 | 593 | 290 | 4.19 |

**특징:**
- 인기 영화들은 대체로 높은 평점 (3.66~4.45)
- 인기도와 품질이 어느 정도 상관관계

---

## 4. 평점(Rating) 분석

### 분포
| 평점 | 개수 | 비율 |
|------|------|------|
| 5.0 | 14,825 | 14.1% |
| 4.5 | 8,174 | 7.8% |
| 4.0 | 28,831 | **27.4%** ← 최빈값 |
| 3.5 | 12,224 | 11.6% |
| 3.0 | 21,676 | 20.6% |
| 2.5 | 5,473 | 5.2% |
| 2.0 | 7,929 | 7.5% |
| 1.5 | 1,564 | 1.5% |
| 1.0 | 3,254 | 3.1% |
| 0.5 | 1,189 | 1.1% |

### 통계량
- **평균**: 3.52
- **중앙값**: 3.5
- **표준편차**: 1.04
- **최빈값**: 4.0

**특징:**
- **긍정 편향(Positive bias)**: 4.0점이 가장 많음
- 3점 이상이 전체의 약 82%
- 부정 평가(≤2.5)는 상대적으로 적음

---

## 5. 이진 분류를 위한 Threshold 분석

추천 여부(O/X)를 결정하기 위한 threshold 검토:

| Threshold | Positive | Negative | 균형도 |
|-----------|----------|----------|--------|
| ≥2.5 | 86.8% | 13.3% | ❌ 매우 불균형 |
| ≥3.0 | 81.5% | 18.5% | ❌ 불균형 |
| ≥3.5 | 60.9% | 39.1% | ⚠️ 다소 불균형 |
| ≥4.0 | 49.3% | 50.7% | ✅ **최적 균형** |
| ≥4.5 | 21.9% | 78.1% | ❌ 매우 불균형 |

### 권장사항
**Threshold = 4.0을 추천**
- Positive/Negative가 거의 50:50으로 균형
- 높은 평점(4점 이상)을 "추천"으로 간주하는 것이 직관적
- 학습 시 class imbalance 문제 최소화

---

## 6. 그래프 구조 분석

### Bipartite Graph 특성
```
User nodes:   668개
Item nodes:   10,321개
Total nodes:  10,989개
Edges:        105,139개
```

### 평균 차수(Degree)
- **User side**: 157.4 (각 사용자가 평균 157개 영화와 연결)
- **Item side**: 10.2 (각 영화가 평균 10명의 사용자와 연결)

### 그래프 특징
1. **매우 희소한 그래프** (98.5% sparsity)
2. **Heterogeneous degree distribution** (사용자와 아이템의 차수 차이 큼)
3. **Long-tail 구조** (소수의 인기 아이템, 다수의 비인기 아이템)

---

## 7. Item ID 특성

### 문제점 발견
- **최소 ID**: 1
- **최대 ID**: 149,532
- **ID 범위**: 149,532
- **실제 고유 아이템 수**: 10,321
- **ID 갭**: 139,211개

### 의미
- Item ID가 **비연속적(non-contiguous)**
- GNN 구현 시 **반드시 re-indexing 필요**
- 0부터 10,320까지 연속적인 인덱스로 매핑 필요

---

## 8. GNN 구현을 위한 도전 과제

### 🚨 주요 도전 과제

**1. 극도로 희소한 데이터 (98.5% sparsity)**
- **문제**: GNN의 message passing이 충분한 정보를 전달하기 어려움
- **해결책**:
  - 적절한 GNN layer 수 선택 (2-3 layer 추천)
  - Negative sampling 전략
  - Graph augmentation 고려

**2. Long-tail 분포**
- **문제**: 79%의 아이템이 10번 이하 상호작용 → Cold-start 문제
- **해결책**:
  - Item feature 추가 (메타데이터 활용 시)
  - Popularity-aware sampling
  - Meta-learning 기법 고려

**3. 불균형한 사용자 활동도**
- **문제**: 사용자별 상호작용 20~5,672개로 매우 다양
- **해결책**:
  - Degree normalization (GraphSAGE의 aggregator)
  - Attention mechanism (GAT)
  - Mini-batch sampling with degree-aware strategy

**4. 비연속적 Item ID**
- **문제**: ID 범위가 크고 갭이 많음
- **해결책**: **필수** re-indexing (User: 0~667, Item: 0~10,320)

**5. 평점의 긍정 편향**
- **문제**: 높은 평점이 많아 negative signal이 부족
- **해결책**:
  - Threshold 4.0 사용하여 균형 맞추기
  - 보지 않은 영화를 implicit negative로 활용
  - BPR Loss 등 ranking loss 사용

---

## 9. 프로젝트 접근 전략 제안

### Phase 1: 데이터 전처리
```python
✓ User ID: 1~668 → 0~667 (re-index)
✓ Item ID: 비연속 → 0~10,320 (mapping dict 생성)
✓ Train/Validation split (80:20 or user-based)
✓ Threshold 4.0 적용 → binary labels
✓ Negative sampling (1:1 or 1:4 ratio)
```

### Phase 2: 그래프 구성
```python
✓ Edge list 생성: (user_idx, item_idx) pairs
✓ PyTorch Geometric format으로 변환
✓ Node features (단순 embedding 또는 one-hot)
```

### Phase 3: GNN 모델 선택

**Option A: LightGCN (추천)** ⭐
- 추천 시스템에 특화된 간단한 GCN
- No feature transformation, No activation
- 빠르고 효과적

**Option B: GraphSAGE**
- Mean/LSTM/Pool aggregator
- 다양한 실험 가능

**Option C: GAT**
- Attention으로 중요한 이웃 강조
- 계산 비용 높음

### Phase 4: 학습 전략
```python
Loss: Binary Cross-Entropy or BPR Loss
Optimizer: Adam (lr=0.001)
Metrics: Precision@K, Recall@K, Hit Rate, AUC
Epochs: 50-100 (early stopping)
Batch size: Full-batch or mini-batch
```

### Phase 5: 평가
```python
✓ Test 파일 형식에 맞게 추론
✓ Threshold 기반 O/X 결정
✓ 최종 정확도 계산
```

---

## 10. 핵심 인사이트 요약

### ✅ 강점
1. **충분한 데이터 양** (105k interactions)
2. **활발한 사용자들** (평균 157개 상호작용)
3. **Threshold 4.0 사용 시 균형잡힌 분류 문제**
4. **GNN의 collaborative filtering 효과 기대**

### ⚠️ 약점
1. **극도의 sparsity** (98.5%)
2. **심각한 long-tail** (79% 아이템이 ≤10 interactions)
3. **사용자 활동도의 높은 분산**
4. **Cold-start 문제**

### 💡 추천 방향

**단계별 접근:**
1. **Baseline**: 간단한 2-layer GCN으로 시작
2. **Improvement 1**: LightGCN 적용
3. **Improvement 2**: Negative sampling 전략 최적화
4. **Improvement 3**: Attention mechanism 추가 (시간 있으면)

**Hyperparameter 우선순위:**
1. Embedding dimension (64 vs 128 vs 256)
2. Number of GNN layers (2 vs 3)
3. Negative sampling ratio (1:1 vs 1:4)
4. Learning rate (0.001 vs 0.01)
5. Dropout (0.0 vs 0.3 vs 0.5)

---

## 11. 예상 시간 배분

| 작업 | 예상 시간 | 우선순위 |
|------|-----------|----------|
| 데이터 전처리 | 2시간 | ⭐⭐⭐ |
| 그래프 구성 | 1시간 | ⭐⭐⭐ |
| Baseline 모델 | 2시간 | ⭐⭐⭐ |
| 학습 & 디버깅 | 3시간 | ⭐⭐⭐ |
| 모델 개선 | 2시간 | ⭐⭐ |
| 추론 & 평가 | 1시간 | ⭐⭐⭐ |
| 결과 시각화 | 1시간 | ⭐⭐ |
| 발표 준비 | 1시간 | ⭐⭐ |

**총 예상 시간**: ~13시간

---

## 12. 다음 단계 (Next Steps)

### 즉시 실행 가능:
1. ✅ **EDA 완료** (현재 단계)
2. ⬜ 데이터 전처리 파이프라인 구축
3. ⬜ PyTorch Geometric 환경 설정
4. ⬜ Baseline GNN 모델 구현
5. ⬜ 학습 루프 작성
6. ⬜ 평가 메트릭 구현
7. ⬜ 추론 스크립트 작성

---

## 결론

이 데이터셋은 전형적인 **추천 시스템 문제**로, GNN을 적용하기에 적합한 구조를 가지고 있습니다.

**핵심 성공 요인:**
- Sparsity 문제를 해결하는 적절한 GNN 설계
- 효과적인 negative sampling
- Long-tail 아이템 처리 전략
- 균형잡힌 binary classification (threshold 4.0)

**최종 목표:**
- Test 데이터에 대해 높은 정확도의 O/X 추천 결과 생성
- Loss curve 시각화
- 모델 구조 문서화
- 5분 발표 준비

---

**Report Generated**: 2025-11-12
**Analysis Tool**: Python (pandas, numpy, matplotlib, seaborn)
