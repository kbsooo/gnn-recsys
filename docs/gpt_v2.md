## 1. 경쟁자 전략들, 뭐가 괜찮고 뭐가 과한지

### 1) claude_v1_gnn.md 요약 & 평가

**좋은 점**

* LightGCN + BPR를 메인으로 잡은 건 나랑 완전히 일치.
* 데이터 통계(희소도, 롱테일, 파워 유저) 분석이 꽤 잘 되어 있고 실제 분포랑도 맞음.
* Negative sampling, hard negative, coverage 같은 실전적인 포인트들이 들어가 있음.
* Baseline(MF, popularity, userKNN)과 비교하자는 마인드도 좋음.

**과한/위험한 부분**

* 10-core filtering(상호작용 10개 미만 유저/아이템 제거) 같은 정제는
  “실제 평가용 test에 그런 애들이 나오면?”을 생각하면 메인 플로우로 두기 위험.
* item-item / user-user 그래프 추가, graph augmentation 등은
  이 데이터 규모에서 “노력 대비 gain”이 애매할 가능성이 큼.
* rating threshold를 3.5로 두고 “추천으로 본다”는 선택은
  mean이 약 3.52라서, 사실상 평균 이상이면 다 추천이라는 뜻이라 좀 느슨함.

### 2) claude_v1_non_gnn.md 요약 & 평가

**좋은 점**

* NeuMF(= GMF + MLP) 를 비-GNN 메인으로 두고

  * MF → NCF/NeuMF → GNN 순으로 난이도/성능 스택 쌓자는 구조는 합리적.
* BPR / BCE 둘 다 고려하고, negative sampling도 정석적으로 설명.
* Non-GNN vs GNN의 trade-off(속도, 구현 난이도, 성능)를 꽤 솔직하게 정리.

**아쉬운 점**

* “LightGCN이 5~10% 정도 항상 더 좋다” 식으로 꽤 낙관적인 가정을 함.

  * 이 정도 사이즈/노이즈면 진짜로는 **MF/NeuMF가 LightGCN 이길 수도 있음**.
* AutoEncoder, DCN, Wide&Deep까지 다 끌고 오는데

  * 이 프로젝트 스코프에서 전부 돌리기엔 실속이 떨어짐.

### 3) grok_v1.md 요약 & 평가

**좋은 점**

* 2025 RecSys 트렌드(contrastive, multi-scale attention, fairness 등)를 잘 끌어옴.
* LightGCN + contrastive + multi-scale attention = “실험용 breakthrough” 방향으로는 가치 있음.

**문제/허언 느낌 나는 부분**

* SPT-GNN, IReGNN, GR-MC 등 최신 논문들을 막 가져오는데

  * 이 과제 규모/시간에 맞는 현실적인 선택지는 아님.
  * “논문 이름만 잔뜩”이 되는 위험이 큼.
* 10-core filtering, 시간 패턴(seasonality) 언급 등 실제 데이터에 없는 정보(타임스탬프)를 가정하는 느낌도 있음.
* threshold 3.5 고정, edge weighting 등 세팅이 근거라기보다 그냥 “그럴싸한 선택” 수준.

---

## 2. 내 기존 생각 vs 경쟁자들 비교해서 정리

내가 처음 제안했던 핵심:

* **Main GNN**: LightGCN + BPR
* **Labeling**: rating ≥ 4.0을 positive로 보는 binary / implicit 혼합.
* **Baselines**: popularity, MF, NeuMF 정도는 필수.
* **Breakthrough**: hard negative, contrastive, multi-task를 “나중에 추가”.

경쟁자들까지 합쳐 보면:

* LightGCN + BPR라는 큰 축은 **다들 동의**.
* Non-GNN 쪽에서는 NeuMF(= NCF 변형)를 **사실상 표준 baseline**으로 보고 있음.
* Threshold 3.5 vs 4.0, 10-core filtering, graph augmentation, multi-scale attention 같은 건 **합의가 없다**. 취향/논문빨에 가까움.

그래서 “최고 전략”은:

> **실제로 구현 가능하고, 리스크/리워드가 좋은 것만 가져오고**
> 논문 쇼케이스용 기능은 후순위 옵션으로 내려버리는 게 맞다.

이 관점으로 다시 짠 최종 전략 문서를 아래처럼 정리할게.

---

## 3. 최종 전략 문서 (GNN + Non-GNN 통합판)

### 3.1 문제 정의 & 데이터 현실

* train.csv:

  * 유저 수: 668
  * 아이템 수: 10,321
  * 상호작용 수: 105,139
  * 밀도: 약 1.52% (전체 user-item 조합 중 98.5%가 비어 있음)
* user별 상호작용 개수:

  * min 20, median 70, max 5672 (파워 유저 존재)
* item별 상호작용 개수:

  * median 3, 75%가 8 이하 → 극단적인 롱테일
* rating 분포:

  * 평균 ≈ 3.52
  * `≥ 4.0`: 약 49.3%
  * `≥ 3.5`: 약 60.9%

**최종 태스크**

* 입력: `(user, item)` 쌍
* 출력: `recommend: O/X`

  * * 마지막 줄에 `Total recommends = a/b, not recommend = c/b`

**중요 포인트**

* rating은 **도움 되는 힌트**일 뿐, 최종 채점은 O/X binary.
* test에 low-frequency user/item이 나올 수 있어서
  **과 aggressive한 filtering(10-core 등)은 메인 전략에서 제외**.

---

## 4. 라벨링 & 데이터 전처리 전략

### 4.1 Positive 정의 (Labeling)

여기서 경쟁자들 의견이 갈리는 부분이라, 아예 **실험 가능한 plan**으로 정리한다.

* Candidate 1: `rating ≥ 4.0` → positive

  * 장점: “좋았다”에 더 확실히 해당하는 샘플만 사용.
  * label 비율 ≈ 49:51 → 거의 balanced.
* Candidate 2: `rating ≥ 3.5` → positive

  * 장점: 샘플 수 증가, 학습이 더 안정적일 수도 있음.
  * 단점: 평균보다 살짝 높다는 이유만으로 positive로 묶어버리는 셈.

**전략**

* **Train/Val split 고정한 뒤**

  * 두 label 정의에 대해 같은 모델(LightGCN, NeuMF)을 돌려서
  * Recall@20 / NDCG@20 / Validation F1(이진 O/X) 비교.
* 실제 숫자로 봐서 더 좋은 쪽을 **최종 라벨 정의로 채택**.

즉, 여기선 “4.0이 더 타당해 보인다”가 아니라,
**threshold 자체를 hyperparameter로 본다.**

### 4.2 ID 매핑

공통 인프라:

1. `user` → `uid ∈ [0, n_users-1]`
2. `item` → `iid ∈ [0, n_items-1]`
3. 딕셔너리 두 개 저장:

   * `user2idx`, `idx2user`
   * `item2idx`, `idx2item`

GNN/Non-GNN 전부 이 매핑 위에서 돌리고,
최종 inference 때 다시 원래 id로 되돌려 출력.

### 4.3 Train / Validation split

* timestamp 없으니 **per-user stratified random split**:

  * 각 user u에 대해:

    * u의 interaction 중 80% → train
    * 나머지 20% → validation candidate
  * validation positive는 **선택한 label 규칙(예: rating ≥ 4)**만 사용.
  * user마다 validation에 최소 1개 이상 positive가 있도록 강제.

* validation negative:

  * user u가 train/val에서 한 번도 보지 않은 item들 중 랜덤 샘플링 (예: 100개).
  * 이걸로 per-user ranking metric 계산.

### 4.4 Negative sampling (train용)

* 기본: random negative sampling

  * 각 positive (u, i⁺)마다

    * u가 보지 않은 item들에서 1~K개 (K=1~4) 샘플 → i⁻.
* 나중에 성능 plateau되면 → **hard negative mining** 추가 (Breakthrough 파트에서).

### 4.5 “정제/필터링”에 대한 입장

* **메인 플로우**에서는:

  * 10-core 같은 aggressive filtering **하지 않는다**.
  * 이유: test 쪽에 low-frequency item이 나오면 완전 cold-start가 되기 때문.
* 다만, 실험용으로:

  * overfitting 분석을 위해 “10-core 버전에서만 학습해본 모델”을 **참고용**으로 추가하는 건 OK.

---

## 5. Non-GNN 딥러닝 전략 (Baseline + Reality Check)

### 5.1 왜 Non-GNN을 진지하게 봐야 하는가

* 이 데이터 사이즈(668 × 10,321 × 105k)면

  * 잘 튜닝된 **MF/NeuMF가 LightGCN 이기는 그림**도 충분히 가능.
* GNN이 이기더라도 “얼마나” 이기는지 숫자로 보여줘야
  “GNN 쓴 의미”가 생긴다.

### 5.2 Baseline 계층 구조

1. **Popularity**

   * item별 interaction 수, 혹은 평균 rating 기준으로 인기 순 정렬.
   * user 무관 global rank.
2. **MF + BPR**

   * user/item embedding만 있고 inner product로 score.
   * BPR loss 사용.
3. **NeuMF (NCF 변형)**

   * GMF + MLP 두 branch 결합.
   * main Non-GNN 모델.

이 셋은 **반드시 돌리고**,
LightGCN 결과를 이들과 **같은 평가 프로토콜로 비교**한다.

### 5.3 MF + BPR

* user embedding `p_u ∈ R^d`
* item embedding `q_i ∈ R^d`
* score: `s(u,i) = <p_u, q_i>`
* loss: BPR (positive = label 1, negative = 샘플)

장점:

* 구현 단순, 학습 빠름.
* GNN/NeuMF 구현 전에 전체 pipeline(negative sampling, evaluation, threshold)이 정상 동작하는지 검증용으로 최적.

### 5.4 NeuMF (추천 Non-GNN 메인)

구조는 경쟁자 문서의 틀을 거의 그대로 가져가되, **불필요하게 깊게 안 간다**.

* GMF 파트:

  * `user_emb_gmf`, `item_emb_gmf` → element-wise product → `z_gmf`.
* MLP 파트:

  * `user_emb_mlp`, `item_emb_mlp` → concat → 2~3 layer MLP → `z_mlp`.
* 최종:

  * `z = concat(z_gmf, z_mlp)` → FC → scalar score.

손실:

* 메인: BPR (GNN과 맞추기)

  * positive: label 1
  * negative: 샘플링한 unobserved item
* 보조 옵션: BCE + sampled negative (성능 비교용)

하이퍼파라미터 추천 범위:

* embedding dim: 64
* hidden layers: [128, 64, 32] 또는 [128, 64]
* dropout: 0.2~0.3
* lr: 1e-3
* weight decay: 1e-5

**Non-GNN 관련 최종 목표**

* MF, NeuMF 모두에 대해:

  * Recall@20, NDCG@20, Validation F1을 계산.
  * LightGCN 대비 **얼마나 차이 나는지 수치화**.

---

## 6. GNN 메인 전략 – LightGCN 중심

### 6.1 모델 선택

* 메인 GNN: **LightGCN**

  * 이유:

    * 이 데이터 크기에 과도한 복잡성(Attention, multi-scale 등)은 오히려 독.
    * 논문/실무에서도 MF + LightGCN 정도가 가장 안정적인 선택.

* NGCF, GAT, multi-scale attention, SPT-GNN 같은 건

  * **“시간 남으면 해보는 실험용”**으로만 고려.

### 6.2 LightGCN 구체 구조

* 노드:

  * user: 0 ~ n_users-1
  * item: n_users ~ n_users + n_items - 1
* 초기 임베딩:

  * `E^0_user ∈ R^{n_users × d}`
  * `E^0_item ∈ R^{n_items × d}`
* adjacency:

  * bipartite 인접행렬 A → 정규화 Â (`D^{-1/2} A D^{-1/2}`)
* layer:

  * `E^{k+1} = Â E^k`
* 최종 임베딩:

  * `E = (E^0 + E^1 + ... + E^K) / (K+1)`
* score:

  * `s(u,i) = <E_u, E_i>`

초기 하이퍼파라미터:

* d: 64
* K: 2 또는 3
* optimizer: AdamW
* lr: 1e-3
* weight decay(L2): 1e-4
* batch size(positive pairs): 1024 ~ 4096

### 6.3 손실 함수 – BPR 단일로 명확하게

* Positive pair (u, i⁺): label 1
* Negative pair (u, i⁻): u가 보지 않았거나 label 0인 item 중 sampling

수식:

* `L = - E[ log σ(s(u, i⁺) - s(u, i⁻)) ] + λ||θ||²`

여기서:

* aux rating regression, multi-task loss(MSE + BPR) 같은 건
  **나중에 ablation용**으로만 추가.

### 6.4 학습 & 안정화 전략

1. **mini-batch BPR**

   * batch 내에 여러 (u, i⁺) pair
   * 각 pair마다 negative 1개(K=1) → 상황보고 K 늘리기.
2. **negative sampling**

   * v1: uniform random from 전체 item
   * v2: popularity-biased(인기 item일수록 자주 negative로) → 나중에 시도.
3. early stopping

   * monitor: Recall@20 (혹은 NDCG@20)
   * patience: 20 epoch 정도.
4. seed, reproducibility

   * seed 고정해서 baseline 성능 재현 가능하게.

### 6.5 Breakthrough 옵션 (우선순위 낮음)

여기부터는 “성능이 plateau 찍은 뒤”에 검토.

1. **hard negative mining**

   * 현재 모델로 score 높은데 실제로는 negative인 item을 뽑아서
     BPR에서 negative로 더 자주 사용.
2. **contrastive / self-supervised**

   * edge dropout, subgraph sampling 등으로 augment된 두 그래프 뽑아서
     같은 노드 embedding끼리 가까워지도록 InfoNCE loss 추가.
3. **edge weight에 rating 반영**

   * rating high edge에 더 큰 weight를 주는 normalized adjacency 실험.

이 세 개 중에서:

* **1번(hard negative)**이 구현 대비 효과가 가장 현실적이고,
* 2,3은 “실험적으로 재미있을 수 있지만 과제 스코프에선 과할 수 있다”.

---

## 7. GNN vs Non-GNN 하이브리드 전략

성능 욕심까지 고려하면:

### 7.1 Score-level Ensemble

* LightGCN score: `s_gnn(u,i)`
* NeuMF score: `s_ncf(u,i)`
* 최종 score:

  * `s_final = α · s_gnn + (1-α) · s_ncf`
  * α는 validation에서 grid search (예: {0.3, 0.5, 0.7})

효과:

* 두 모델이 서로 다른 패턴을 잡기 때문에
  variance 줄이면서 성능 2~5% 정도 더 올릴 수 있는 여지가 있음.

### 7.2 Two-stage (시간 남으면)

* Stage 1 (빠른 모델, 예: NeuMF/Popularity):

  * user당 Top-200 후보 item 추출.
* Stage 2 (LightGCN):

  * 해당 후보에 대해서만 정교하게 score → rerank.

장점:

* full ranking을 매번 전체 item에 대해 하지 않아도 되니 속도 이득.
* 시험 환경이 작은 데이터라 성능 이득은 크지 않을 수 있으나,
  “실제 서비스라면 이렇게 한다”는 설명용으로 쓸 만함.

---

## 8. 평가 & 최종 O/X 결정

### 8.1 Offline metric (model selection용)

Validation에서 per-user 기준:

* Recall@K (K=10, 20)
* NDCG@K (K=10, 20)
* HitRate@K (참고)
* Coverage (얼마나 다양한 item을 추천하는지)

**main metric**은:

* `Recall@20` 혹은 `NDCG@20` 하나를 **주 지표로 고정**.
* 나머지는 부지표로 보고, 성능 비교할 때 항상 이 main metric 중심으로 판단.

### 8.2 O/X를 위한 score threshold 튜닝

1. validation의 (u, i) pair 전부에 대해:

   * 모델 score `s(u,i)`를 구하고
   * label (rating threshold 기반 binary)을 준비.
2. 여러 threshold t 후보에 대해:

   * accuracy, precision, recall, F1 계산.
3. **F1 혹은 balanced accuracy** 최대화하는 t를 고른다.
4. 최종 inference:

   * test 파일의 (u,i)에 대해 score 계산 후

     * `s ≥ t` → `O`
     * `s < t` → `X`
   * unknown user/item:

     * user/item이 train에 없으면 score를 매우 낮은 값으로 강제 → `X` 처리.

### 8.3 출력 포맷

* 각 row: `user item recommend` (recommend ∈ {O, X})
* 마지막 줄:

  * `Total recommends = a/b, not recommend = c/b`

---

## 9. 실험 로드맵 (현실적인 타임라인)

### Phase 0 – 인프라 세팅 (1~2일)

* EDA (지금 한 것처럼 user/item 분포, rating 분포 재확인)
* ID 매핑, train/val split, negative sampling 코드 작성.
* 평가 코드 (Recall@K, NDCG@K, F1, coverage) 구현 & 유닛테스트.

### Phase 1 – Non-GNN 베이스라인 (3~4일)

1. Popularity, MF + BPR 구현 및 검증.
2. NeuMF (BPR) 구현.
3. label threshold(4.0 vs 3.5) 실험.
4. 결과 정리:

   * “Non-GNN만 썼을 때 best config”를 하나 픽스해둔다.

### Phase 2 – LightGCN 메인 (4~6일)

1. LightGCN 구현 (PyG or 직접).
2. 같은 데이터/label/negative sampling/metric으로 Non-GNN과 비교.
3. embedding dim, layer 수, negative ratio, lr 등 간단한 하이퍼 튜닝.
4. “LightGCN vs NeuMF vs MF” 비교 테이블 작성.

### Phase 3 – Breakthrough & Ensemble (옵션, 3~5일)

1. Hard negative mining 적용해본 LightGCN.
2. 필요하면 contrastive loss / edge weighting 소규모 실험.
3. LightGCN + NeuMF ensemble score tuning.
4. 최종 best 모델 및 설정 정리.

### Phase 4 – Inference 파이프라인 & 문서화 (2~3일)

1. 모델 저장/로드, 테스트 파일 읽기 → O/X 출력 코드 구현.
2. corner case (unknown user/item, 빈 row 등) 처리.
3. 프로젝트 리포트에:

   * 데이터 분석
   * 모델별 성능 비교
   * GNN vs Non-GNN 차이
   * 실패 케이스 / 한계점
   * 앞으로 할 수 있는 확장 아이디어
     를 정리.

---

## 10. Devil’s advocate: 이 전략의 리스크도 짚고 가자

* **라벨 threshold 실험 안 하고 4.0만 믿고 가면?**

  * 실제로는 3.5 기준이 성능 더 좋을 수도 있음.
    그래서 threshold를 hyperparameter로 놓고 **수치 비교**하는 게 중요.
* **GNN이 생각보다 이득이 없을 수도 있다.**

  * MF/NeuMF가 비슷하거나 더 잘 나오는 결과가 나와도 이상하지 않음.
    이 경우

    * “왜 그런지”를 데이터/모델 구조 관점에서 설명하는 게 프로젝트 포인트가 된다.
* **hard negative / contrastive 같은 고급 기법**은

  * 구현 난이도 대비 gain이 미미할 수 있고, 쉽게 망가뜨릴 수 있음.
    → baseline 안정화 이전에는 절대 손대지 말 것.


