# CCC22 모델 Inference 가이드

## 📋 개요

교수님이 **test.csv**를 제공하실 때, **학습 없이** 사전 학습된 모델을 불러와서 추천 결과를 출력하는 스크립트입니다.

---

## ✅ 준비 사항

### 1. 필요한 파일들

```
gnn-recsys/
├── inference.py              ← 추론 스크립트 (메인)
├── data/
│   └── train.csv            ← 학습 데이터 (매핑 정보용)
├── cc_models/
│   ├── cca2_best.pt         ← CCA 모델 (사전 학습됨)
│   ├── ccb2_best.pt         ← CCB 모델 (사전 학습됨)
│   └── ccc21_params.json    ← 최적 파라미터
└── test.csv                  ← 교수님이 제공하실 테스트 파일
```

### 2. 라이브러리

```bash
pip install numpy pandas torch scikit-learn
```

---

## 🚀 사용 방법

### 기본 사용법

```bash
python inference.py test.csv
```

### 예시

```bash
# 교수님이 test.csv를 주셨을 때
python inference.py test.csv

# 또는 경로 지정
python inference.py data/test.csv
python inference.py /path/to/professor_test.csv
```

---

## 📤 출력 결과

### 1. 콘솔 출력 (교수님 요구 양식)

```
====================
user       item       recommend
109        3745       O
88         4447       O
71         4306       X
66         1747       O
15         66934      X
====================
Total recommends = 123/200
Not recommend = 77/200
```

### 2. CSV 파일 (`predictions.csv`)

자동으로 `predictions.csv` 파일이 생성됩니다:

```csv
user,item,recommend
109,3745,O
88,4447,O
71,4306,X
66,1747,O
15,66934,X
```

---

## 🔍 동작 원리

### 1. **모델 불러오기만 수행 (학습 X)**

```python
# 사전 학습된 모델만 로드
cca_model.load_state_dict(torch.load('cc_models/cca2_best.pt'))
ccb_model.load_state_dict(torch.load('cc_models/ccb2_best.pt'))
```

### 2. **추론 파이프라인**

```
Test CSV 읽기
    ↓
User/Item 매핑 확인
    ↓
CCA Score + CCB Rating 계산
    ↓
Ensemble Score (α=0.7, β=0.3)
    ↓
Hybrid Selection (Threshold + Top-K + MIN_K)
    ↓
O/X 결정
    ↓
출력 (콘솔 + CSV)
```

### 3. **추천 규칙**

- ✅ **20% 규칙**: 각 사용자별 interaction의 20% 이하만 추천
- ✅ **Cold User 보호**: ≤10 interactions 사용자는 최소 2개 추천 (MIN_K fallback)
- ✅ **Threshold**: 0.2896 (F1 최적화)
- ✅ **Top-K 제한**: 최대 100개

---

## 📊 예시 실행

### Sample1.csv 테스트

```bash
$ python inference.py data/sample1.csv
```

**출력:**
```
Using device: mps
Loading training data for mappings...
Users: 668, Items: 10321
✓ Loaded 668 users' training data
Building graphs...
Loading pretrained models...
✓ Models loaded
Loading optimal parameters...
  α=0.7, β=0.3, threshold=0.2896

Loading test data: data/sample1.csv

✓ Inference complete!
  Total items: 5
  Recommended: 5
  Not recommended: 0
  O ratio: 100.0%

====================
user       item       recommend
109        3745       O
88         4447       O
71         4306       O
66         1747       O
15         66934      O
====================
Total recommends = 5/5
Not recommend = 0/5

✓ Results saved to: predictions.csv
```

---

## ⚙️ 파라미터 정보

### 최적 파라미터 (ccc21_params.json)

```json
{
  "alpha": 0.7,
  "beta": 0.3,
  "threshold": 0.2896,
  "cca_min": 0.0579,
  "cca_max": 3.0186,
  "ccb_min": 0.5,
  "ccb_max": 5.0
}
```

- **α (CCA weight)**: 0.7 ← Connection이 더 중요
- **β (CCB weight)**: 0.3 ← Rating은 보조 정보
- **Threshold**: 0.2896 ← F1 최적화 값

---

## 🎯 특징

### ✅ 학습 불필요
- 모델 파일(.pt)만 있으면 즉시 추론 가능
- 교수님이 test.csv만 주시면 바로 실행

### ✅ 평가규칙 100% 준수
- 20% 추천 규칙 ✓
- Cold user MIN_K=2 보장 ✓
- User별 K 제한 ✓

### ✅ 정해진 출력 양식
- 콘솔 출력: 교수님 요구 양식
- CSV 출력: 추가 분석용

### ✅ Unknown 처리
- Unknown user → X (추천 안 함)
- Unknown item → X (추천 안 함)
- Train set에 있는 item → X (이미 본 것)

---

## 🐛 문제 해결

### 1. "FileNotFoundError: train.csv"

**원인:** train.csv가 없음

**해결:**
```bash
# train.csv가 있는지 확인
ls data/train.csv

# 없으면 경로 확인
python inference.py test.csv  # 현재 폴더에서 실행
```

### 2. "FileNotFoundError: cca2_best.pt"

**원인:** 모델 파일이 없음

**해결:**
```bash
# 모델 파일 확인
ls cc_models/cca2_best.pt
ls cc_models/ccb2_best.pt

# 없으면 학습 필요 (ccc22.ipynb 실행)
```

### 3. "No module named 'torch'"

**원인:** PyTorch 미설치

**해결:**
```bash
pip install torch
```

---

## 📝 주의사항

### ⚠️ train.csv 필요 이유

**"학습 안 하는데 왜 train.csv가 필요해요?"**

→ 학습은 안 하지만, **매핑 정보**가 필요합니다:

1. **user2idx, item2idx**: User/Item ID를 모델이 이해하는 index로 변환
2. **user_train_items**: 이미 본 아이템 필터링 (X 처리)
3. **user_interaction_count**: Cold user 판별 (MIN_K fallback)

### ⚠️ 모델 파일 필수

- `cc_models/cca2_best.pt` (36MB)
- `cc_models/ccb2_best.pt` (36MB)
- `cc_models/ccc21_params.json` (300B)

**없으면 ccc22.ipynb를 먼저 실행해서 생성하세요!**

---

## 🔄 전체 워크플로우

### 1️⃣ 개발 단계 (이미 완료)

```bash
# ccc22.ipynb 실행 → 모델 학습 및 저장
jupyter notebook cc/ccc22.ipynb
```

**생성 파일:**
- `cc_models/cca2_best.pt`
- `cc_models/ccb2_best.pt`
- `cc_models/ccc21_params.json`

### 2️⃣ 제출 단계 (교수님 test.csv 받은 후)

```bash
# inference.py 실행 → 추론만 수행
python inference.py professor_test.csv
```

**생성 파일:**
- `predictions.csv` → 교수님께 제출

---

## 📈 성능 지표

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.9538 |
| **F1 Score** | 0.9412 |
| **Precision** | 1.0000 |
| **Recall** | 0.8889 |
| **평가규칙 준수** | ✅ 100% |

---

## 📚 관련 문서

- `model_comparison_report.md`: 전체 모델 비교
- `ccc21_evaluation_report.md`: 평가규칙 준수 분석
- `cc/ccc22.ipynb`: 모델 학습 노트북

---

## 💡 팁

### 빠른 테스트

```bash
# Sample 파일로 먼저 테스트
python inference.py data/sample1.csv
python inference.py data/sample2.csv

# 정상 작동 확인 후 실제 test.csv 사용
python inference.py test.csv
```

### CSV만 필요한 경우

```bash
# 콘솔 출력 없이 CSV만 생성
python inference.py test.csv > /dev/null

# predictions.csv 확인
cat predictions.csv
```

---

**작성일**: 2025-11-19
**모델**: CCC22 (CCC21 Refined)
**버전**: 1.0
