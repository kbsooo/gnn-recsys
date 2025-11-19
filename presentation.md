---
marp: true
theme: gaia
_class: lead
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
style: |
  section {
    font-size: 24px;
    text-align: left;
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
  }
  h1 {
    font-size: 48px;
    color: #2c3e50;
  }
  h2 {
    font-size: 36px;
    color: #34495e;
  }
---

# GNN 기반 추천 시스템
## 그래프 신경망과 빅데이터 프로젝트

---

# 1. 데이터셋 및 전처리

- **데이터**: 유저-영화 상호작용 데이터.
- **그래프 구성**:
  - **노드**: 유저($N$) + 아이템($M$).
  - **엣지**: 상호작용 이력을 바탕으로 생성.
- **데이터 분할 전략 (Splitting)**:
  - **계층적 분할 (Stratified Split)**: 모든 유저가 Train/Val/Test에 포함되도록 보장.
  - **평점 로직**:
    - **Bad ratings (< 4.0)**: 구조 학습 및 부정적 선호 학습을 위해 항상 **Train** 셋에 포함.
    - **Good ratings (>= 4.0)**: 70/15/15 비율로 Train/Val/Test 분할.

---

# 2. 방법론: 이원화 접근 (Dual-Version)

데이터의 서로 다른 측면을 포착하기 위해 두 가지 버전을 구현했습니다:

### **Ver A: 구조적 접근 (Structural/Implicit)**
- **개념**: 평점 값을 무시하고 연결의 존재 여부만 긍정으로 간주.
- **그래프**: 비가중치(Unweighted) 인접 행렬.
- **목표**: 이진 분류 (연결 됨 vs 안 됨).

### **Ver B: 선호도 접근 (Preference/Explicit)**
- **개념**: 평점 값을 반영하여 선호도 차이를 학습.
- **그래프**: 가중치(Weighted) 인접 행렬 (평점 기반 가중치).
- **임계값**: 평점 4.0 이상을 "True Positive"로 간주.
- **목표**: 평점 예측 및 선호도 랭킹.

---

# 3. 모델 아키텍처: LightGCN

협업 필터링(Collaborative Filtering)에 특화된 **LightGCN**을 사용했습니다.

- **단순화된 설계**: 과적합 방지를 위해 불필요한 특징 변환 및 비선형 활성화 함수 제거.
- **임베딩 전파 (Propagation)**:
  $$E^{(k+1)} = D^{-1/2} A D^{-1/2} E^{(k)}$$
- **레이어 통합 (Aggregation)**:
  $$E_{final} = \text{Mean}(E^{(0)}, E^{(1)}, \dots, E^{(K)})$$
- **Dual Heads**:
  - **A-Model**: 구조적 점수 계산 (Dot Product).
  - **B-Model**: 평점 예측을 위한 MLP Head.

---

# 4. 학습 전략

### 손실 함수 (Loss Functions)
- **Weighted BPR Loss**: 평점 신뢰도를 가중치로 반영한 베이지안 개인화 랭킹 손실.
- **MSE Loss**: 모델 B의 평점 예측 정확도 향상을 위해 사용.

### 네거티브 샘플링 (Negative Sampling)
- **Hard Negative Sampling**:
  - 모델이 잘못 랭킹한(높은 점수를 준) 부정 아이템을 동적으로 샘플링.
  - 유저 선호도의 미세한 차이를 학습하도록 유도.

---

# 5. 앙상블 및 최적화 (Ensemble)

성능 극대화를 위해 두 모델을 결합했습니다:

1. **강건한 정규화 (Robust Normalization)**: 
   - 백분위수 기반 Min-Max 스케일링을 통해 점수 분포 차이 보정.
2. **선형 결합**:
   $$S_{final} = \alpha \cdot S_{A} + (1-\alpha) \cdot S_{B}$$
3. **그리드 서치 (Grid Search)**:
   - 검증(Validation) 셋을 통해 최적의 $\alpha$ 탐색.
   - **최적 결과**: $\alpha = 0.7$ (구조 70%, 평점 30% 반영).

---

# 6. 추천 로직 (Recommendation Logic)

프로젝트 요구사항에 맞춘 동적 추천 규칙을 구현했습니다:

- **Top-K 및 임계값 (Thresholding)**:
  - 예측 점수가 임계값을 넘는 아이템만 추천.
  - **제약 조건**: 기존 상호작용 수의 20% 이하 (또는 선호 아이템의 50%) 개수까지만 추천.
  - **콜드 스타트 (Cold-Start)**: 상호작용이 10개 이하인 유저는 고정된 개수(예: 2개) 추천.

---

# 7. 평가 결과 (Evaluation Results)

최종 테스트 셋(Test Set) 성능 지표:

| 지표 (Metric) | 점수 (Score) | 해석 |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.9332** | 탁월한 랭킹 성능 입증. |
| **F1 Score** | **0.8708** | 정밀도(Precision)와 재현율(Recall)의 높은 균형. |
| **Accuracy** | **0.8730** | 상호작용 여부를 정확하게 분류. |
| **O-Ratio** | **0.4830** | - |

---

# 8. 한계점 및 고찰 (Limitations)

- **순차적 필터링의 딜레마**:
  - **Ver A (구조)** 단계에서 1차 필터링 후, **Ver B (선호도)** 단계에서 최종 추천 여부를 결정하는 방식.
  - **장점**: 확실하지 않은 아이템을 배제하여 정밀도(Precision)를 높임.
  - **단점**: 두 단계의 엄격한 기준을 모두 통과해야 하므로, 최종적으로 **추천되는 아이템의 개수가 다소 적어지는 경향**이 있음.
  - **향후 과제**: Recall을 보완하기 위한 필터링 기준 완화 또는 추가적인 탐색(Exploration) 기법 도입 고려 필요.

---

# 9. 결론 (Conclusion)

- **그래프의 힘**: GNN을 통해 유저-영화 간의 잠재적 구조를 성공적으로 포착했습니다.
- **하이브리드 접근**: 문제를 구조(A)와 선호도(B)로 분리하여 각각 전문화된 학습을 수행했습니다.
- **높은 정확도**: 앙상블 모델을 통해 **AUC 93% 이상**을 달성하며 제안된 방법론의 유효성을 입증했습니다.
