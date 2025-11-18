 데이터 분석/정제 아이디어

  - 기본 통계: 유저/아이템 수, 상호작용 수, sparsity, 유저·아이템 당 평균/분포, long-tail 정도
    확인. rating 분포(특히 4 기준)와 시간 컬럼 존재 여부 확인.
  - 이상치/노이즈: 중복 상호작용, 유저·아이템 cold-start 비율. train/val/test 스플릿 시 cold
    user/item 처리 전략 명시.
  - Negative sampling: A버전은 미관측=잠재적 음성으로 간주, popular/stratified 방식으로 샘플링.
    B버전은 rating<4를 음성으로 활용 + 미관측 샘플을 섞는 혼합 전략 검토.
  - Split 전략: 유저별 time-based holdout(가능하다면) vs random split. 평가 재현성을 위해 seed
    고정.

  임베딩/모델 설계

  - 공통: user/item embedding + message passing(GNN). GIN/GAT/LightGCN 후보. LightGCN은 단순/효
    율, GAT는 attention으로 가중치 학습. 두 버전 모두 edge weight에 rating을 쓸지 여부만 차이.
  - A버전: unweighted bipartite graph. LightGCN + BPRLoss or BCE(from negatives). Edge dropout/
    embedding norm regularization. Popularity-bias 보정 위해 debiasing negative sampling 가능.
  - B버전: weighted graph (rating 스케일 반영). 두 접근:
      1. 이진화(>=4 pos, <4 neg)로 A버전과 동일 파이프라인.
      2. 점수 예측: GNN 후 MLP로 rating regression (MSE/MAE) or ordinal classification. 추론 시
         threshold=4로 recommend.
  - Side info 부재 시 regularization, dropout, multi-hop propagation depth 튜닝(2~3 hop)으로 성
    능 확보. 임베딩 차원 64~128 시도.

  - 새로운 평가규칙 반영: 유저별 추천 수 제한(top-K, threshold). 예측 스코어 정렬 후 규칙에 맞게
    컷오프(K=⌊0.2·pos_count⌋, 최소 2개) 적용한 Hit/Recall 측정.

  전처리/파이프라인 설계

  - ID 인코딩: 유저/아이템 integer mapping. sparse CSR 구축.
  - Negative sampling: epoch별 on-the-fly 생성. popular/uniform 믹스.
  - Graph build: bi-partite edge list, optionally edge weight=rating. self-loop 불필요(특
    히 LightGCN).
  - Split: train/val/test by user, ratio 예: 8/1/1. Time 정보 있으면 latest holdout.
  - Batch: neighbor aggregation by propagation precompute(LightGCN adjacency normalization).
    Mini-batch BPR triplets.

  출력/테스트 플로우

  - 모델 저장: @codex_models/에 버전별 파일(pth).
  - 추론 스크립트(notebook 내부 함수)로 임의 CSV(user,item[,rating]) 받아 recommend O/X 출력 및
    총합 집계.
  - 샘플: sample1.csv(A=OOOOO, B=XXOXO 예상) 검증 로직 넣기.

  브레이크스루/실험 아이디어

  - Two-tower vs LightGCN 비교: two-tower dot-product + in-batch negatives, 더 빠른 베이스라인.
  - Popularity-aware sampling: 드문 아이템 강조.
  - Adaptive threshold: 유저별 score 분포 기반 퍼센타일 컷(규칙 범위 내)로 추천 개수 조정.
  - Embedding regularization: L2 + dropout on propagation.
  - Curriculum negatives: early epochs easy negatives(unobserved), later hard negatives(popular
    or user-history-adjacent).

  리포 구조 제안

  - @codex/codexa1.ipynb: A버전 구현·실험.
  - @codex/codexb1.ipynb: B버전 구현·실험.
  - @codex_docs/: 데이터 통계, 실험 로그, 평가 방법, 사용법 정리(md).
  - @codex_models/: 저장된 체크포인트, 버전명 포함.