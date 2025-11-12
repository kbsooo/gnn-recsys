# 🎯 EDA Quick Summary

## 데이터셋 한눈에 보기

### 📊 기본 통계
```
Users:        668명
Items:        10,321개
Interactions: 105,139개
Sparsity:     98.48%
Density:      1.52%
```

### 👥 사용자 특성
```
평균 상호작용:    157개
중앙값:           70개
활동범위:         20 ~ 5,672개
평균 평점:        3.66 (std: 0.46)
```

### 🎬 영화 특성
```
평균 상호작용:    10개
중앙값:           3개
인기범위:         1 ~ 324개

Long-tail:
  - 1회만:        35.3%
  - ≤5회:         67.4%
  - ≤10회:        79.1%
```

### ⭐ 평점 분포
```
최빈값:    4.0 (27.4%)
평균:      3.52
중앙값:    3.5

긍정편향:  82%가 3점 이상
```

### 🎯 Threshold 분석 (추천 O/X)
```
≥4.0: 49.3% vs 50.7%  ← ✅ 추천!
≥3.5: 60.9% vs 39.1%
≥3.0: 81.5% vs 18.5%
```

---

## 🚨 핵심 도전 과제

1. **극도의 Sparsity (98.5%)**
   - GNN message passing 제한적

2. **Long-tail 분포**
   - 79% 아이템이 ≤10 interactions
   - Cold-start 문제

3. **사용자 활동도 분산**
   - 20 ~ 5,672개 (차이 284배)
   - Degree normalization 필요

4. **비연속적 Item ID**
   - 1 ~ 149,532 범위, 실제 10,321개
   - Re-indexing 필수

5. **평점 긍정 편향**
   - Negative sampling 전략 중요

---

## 💡 추천 전략

### 전처리
- [x] User/Item re-indexing (0부터 시작)
- [x] Threshold 4.0 사용
- [x] Train/Val split (80:20)
- [x] Negative sampling (1:1 ratio)

### 모델
- [x] **LightGCN** (1순위) - 간단하고 효과적
- [ ] GraphSAGE (2순위) - 다양한 aggregator
- [ ] GAT (3순위) - Attention (복잡함)

### 학습
- [x] Loss: Binary Cross-Entropy or BPR
- [x] Epochs: 50-100 (early stopping)
- [x] Embedding dim: 64 or 128
- [x] Layers: 2-3
- [x] Metrics: Precision@K, Recall@K, Hit Rate

---

## 📈 기대 효과

### GNN이 도움 될 것:
✅ Collaborative filtering (유사 사용자/아이템 발견)
✅ Sparse 데이터에서 패턴 학습
✅ Multi-hop neighbor 정보 활용
✅ Implicit feedback 처리

### 주의할 점:
⚠️ Over-smoothing (layer 너무 많으면 성능 저하)
⚠️ Long-tail item 성능 낮을 수 있음
⚠️ Negative sampling 전략에 따라 성능 차이 큼

---

## 📁 생성된 파일

```
results/
├── EDA_REPORT.md                    (상세 분석 리포트)
├── QUICK_SUMMARY.md                 (이 파일)
├── eda_visualizations.png           (9개 시각화)
└── eda_detailed_analysis.png        (3개 심화 분석)

notebooks/
├── eda_analysis.py                  (통계 분석 스크립트)
└── visualizations.py                (시각화 스크립트)
```

---

## 🎯 다음 스텝

1. ⬜ **데이터 전처리 구현** (preprocessing.py)
2. ⬜ **PyTorch Geometric 설치**
3. ⬜ **Baseline GNN 모델** (model.py)
4. ⬜ **학습 루프** (train.py)
5. ⬜ **추론 스크립트** (inference.py)
6. ⬜ **평가 & 시각화**
7. ⬜ **발표 준비**

---

**Ready to Code? Let's build the GNN! 🚀**
