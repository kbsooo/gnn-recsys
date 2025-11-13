"""
Breakthrough Analysis for GNN-RecSys
분석 목표: 왜 성능이 낮은지, 어떤 breakthrough가 필요한지 파악
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 12)

# 데이터 로드
data_dir = '../data/processed'

# ID mappings
with open(os.path.join(data_dir, 'id_mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)

train_df = pd.read_csv(os.path.join(data_dir, 'train_split_v3.csv'))
valid_df = pd.read_csv(os.path.join(data_dir, 'valid_split_v3.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_split_v3.csv'))

print("=" * 80)
print("BREAKTHROUGH ANALYSIS: 왜 성능이 낮은가?")
print("=" * 80)

# 1. 기본 통계
print("\n[1] 기본 데이터 통계")
print("-" * 80)
print(f"Train: {len(train_df):,} samples ({len(train_df)/len(pd.concat([train_df, valid_df, test_df]))*100:.1f}%)")
print(f"Valid: {len(valid_df):,} samples ({len(valid_df)/len(pd.concat([train_df, valid_df, test_df]))*100:.1f}%)")
print(f"Test:  {len(test_df):,} samples ({len(test_df)/len(pd.concat([train_df, valid_df, test_df]))*100:.1f}%)")
print(f"\nTotal users: {len(mappings['user_id_map'])}")
print(f"Total items: {len(mappings['item_id_map'])}")

# 2. Cold-Start 분석
print("\n[2] Cold-Start 아이템 분석")
print("-" * 80)

train_items = set(train_df['item_id'].unique())
valid_items = set(valid_df['item_id'].unique())
test_items = set(test_df['item_id'].unique())
train_valid_items = train_items | valid_items

test_cold_items = test_items - train_valid_items
test_warm_items = test_items & train_valid_items

print(f"Train에만 있는 아이템: {len(train_items - test_items):,}")
print(f"Test에만 있는 아이템 (완전 cold): {len(test_cold_items):,}")
print(f"Train+Valid와 겹치는 Test 아이템: {len(test_warm_items):,}")
print(f"\nTest의 cold-start 비율: {len(test_cold_items)/len(test_items)*100:.2f}%")

# Test에서 cold-start 아이템이 포함된 샘플 수
test_cold_samples = test_df[test_df['item_id'].isin(test_cold_items)]
print(f"Test 샘플 중 cold-start 아이템 포함: {len(test_cold_samples):,} / {len(test_df):,} ({len(test_cold_samples)/len(test_df)*100:.2f}%)")

# 3. Item 인기도 분석
print("\n[3] 아이템 인기도 분석 (Train에서의 등장 횟수)")
print("-" * 80)

train_item_counts = train_df['item_id'].value_counts()
test_item_popularity = []

for item_id in test_df['item_id'].unique():
    if item_id in train_item_counts.index:
        test_item_popularity.append(train_item_counts[item_id])
    else:
        test_item_popularity.append(0)

test_item_popularity = np.array(test_item_popularity)

print(f"Test 아이템의 Train 등장 횟수:")
print(f"  - 평균: {test_item_popularity.mean():.2f}")
print(f"  - 중앙값: {np.median(test_item_popularity):.2f}")
print(f"  - 최소: {test_item_popularity.min()}")
print(f"  - 최대: {test_item_popularity.max()}")
print(f"  - 표준편차: {test_item_popularity.std():.2f}")

# Long-tail 분포
print(f"\nTest 아이템 중:")
print(f"  - Train에 0번 등장 (cold): {(test_item_popularity == 0).sum()} ({(test_item_popularity == 0).sum()/len(test_item_popularity)*100:.1f}%)")
print(f"  - Train에 1-5번: {((test_item_popularity >= 1) & (test_item_popularity <= 5)).sum()} ({((test_item_popularity >= 1) & (test_item_popularity <= 5)).sum()/len(test_item_popularity)*100:.1f}%)")
print(f"  - Train에 6-10번: {((test_item_popularity >= 6) & (test_item_popularity <= 10)).sum()} ({((test_item_popularity >= 6) & (test_item_popularity <= 10)).sum()/len(test_item_popularity)*100:.1f}%)")
print(f"  - Train에 11-50번: {((test_item_popularity >= 11) & (test_item_popularity <= 50)).sum()} ({((test_item_popularity >= 11) & (test_item_popularity <= 50)).sum()/len(test_item_popularity)*100:.1f}%)")
print(f"  - Train에 50번 이상: {(test_item_popularity > 50).sum()} ({(test_item_popularity > 50).sum()/len(test_item_popularity)*100:.1f}%)")

# 4. User 활동도 분석
print("\n[4] 사용자 활동도 분석")
print("-" * 80)

train_user_counts = train_df['user_id'].value_counts()
test_user_counts = test_df['user_id'].value_counts()

print(f"Train - User당 평균 상호작용: {train_user_counts.mean():.2f}")
print(f"Test - User당 평균 상호작용: {test_user_counts.mean():.2f}")
print(f"\nTrain - User 상호작용 범위: {train_user_counts.min()} ~ {train_user_counts.max()}")
print(f"Test - User 상호작용 범위: {test_user_counts.min()} ~ {test_user_counts.max()}")

# 5. Rating 분포 비교
print("\n[5] Rating 분포 비교")
print("-" * 80)

train_rating_dist = train_df['rating'].value_counts(normalize=True).sort_index()
test_rating_dist = test_df['rating'].value_counts(normalize=True).sort_index()

print(f"Train Rating 분포:")
for rating, pct in train_rating_dist.items():
    print(f"  {rating}: {pct*100:.2f}%")

print(f"\nTest Rating 분포:")
for rating, pct in test_rating_dist.items():
    print(f"  {rating}: {pct*100:.2f}%")

print(f"\nTrain 평균 Rating: {train_df['rating'].mean():.3f}")
print(f"Test 평균 Rating: {test_df['rating'].mean():.3f}")

# 6. Graph 연결성 분석
print("\n[6] Graph 연결성 분석")
print("-" * 80)

# User별 unique item 수
train_user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
test_user_items = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

# Test user가 Train에서 본 아이템과의 중복
overlap_ratios = []
for user_id in test_user_items.keys():
    train_items_set = train_user_items.get(user_id, set())
    test_items_set = test_user_items[user_id]

    # 이 user의 test 아이템 중 train에 등장한 아이템 비율
    test_items_in_train = sum(1 for item in test_items_set if item in train_item_counts.index)
    overlap_ratios.append(test_items_in_train / len(test_items_set))

print(f"Test user의 아이템이 Train 전체에 등장한 비율:")
print(f"  - 평균: {np.mean(overlap_ratios)*100:.2f}%")
print(f"  - 중앙값: {np.median(overlap_ratios)*100:.2f}%")

# 7. Sparsity 분석
print("\n[7] Sparsity 분석")
print("-" * 80)

n_users = len(mappings['user_id_map'])
n_items = len(mappings['item_id_map'])

train_sparsity = 1 - (len(train_df) / (n_users * n_items))
test_sparsity = 1 - (len(test_df) / (n_users * n_items))

print(f"Train sparsity: {train_sparsity*100:.4f}%")
print(f"Test sparsity: {test_sparsity*100:.4f}%")
print(f"Overall sparsity: {(1 - (len(train_df) + len(test_df)) / (n_users * n_items))*100:.4f}%")

# 8. 시각화
print("\n[8] 시각화 생성 중...")
print("-" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8-1. Test 아이템의 Train 등장 횟수 분포
ax1 = fig.add_subplot(gs[0, 0])
bins = [0, 1, 5, 10, 20, 50, 100, max(test_item_popularity)+1]
ax1.hist(test_item_popularity, bins=bins, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Train에서의 등장 횟수')
ax1.set_ylabel('Test 아이템 수')
ax1.set_title('Test 아이템의 Train 인기도 분포')
ax1.set_xscale('symlog')
ax1.grid(True, alpha=0.3)

# 8-2. Rating 분포 비교
ax2 = fig.add_subplot(gs[0, 1])
ratings = sorted(set(train_df['rating'].unique()) | set(test_df['rating'].unique()))
train_counts = [train_rating_dist.get(r, 0) for r in ratings]
test_counts = [test_rating_dist.get(r, 0) for r in ratings]
x = np.arange(len(ratings))
width = 0.35
ax2.bar(x - width/2, train_counts, width, label='Train', alpha=0.7)
ax2.bar(x + width/2, test_counts, width, label='Test', alpha=0.7)
ax2.set_xlabel('Rating')
ax2.set_ylabel('비율')
ax2.set_title('Train vs Test Rating 분포')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{r:.1f}' for r in ratings])
ax2.legend()
ax2.grid(True, alpha=0.3)

# 8-3. User 활동도 비교
ax3 = fig.add_subplot(gs[0, 2])
ax3.boxplot([train_user_counts.values, test_user_counts.values],
            labels=['Train', 'Test'])
ax3.set_ylabel('User당 상호작용 수')
ax3.set_title('User 활동도 비교')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 8-4. Cold-start 샘플 비율
ax4 = fig.add_subplot(gs[1, 0])
labels = ['Warm Items', 'Cold Items']
sizes = [len(test_df) - len(test_cold_samples), len(test_cold_samples)]
colors = ['#66b3ff', '#ff9999']
ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Test Set: Cold vs Warm Items')

# 8-5. Item popularity distribution (log scale)
ax5 = fig.add_subplot(gs[1, 1])
train_pop_sorted = sorted(train_item_counts.values, reverse=True)
ax5.plot(train_pop_sorted, linewidth=2)
ax5.set_xlabel('Item Rank')
ax5.set_ylabel('등장 횟수 (log scale)')
ax5.set_title('Item 인기도 Long-tail 분포 (Train)')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)

# 8-6. Test 아이템 카테고리별 개수
ax6 = fig.add_subplot(gs[1, 2])
categories = ['Cold\n(0회)', '1-5회', '6-10회', '11-50회', '50+회']
counts = [
    (test_item_popularity == 0).sum(),
    ((test_item_popularity >= 1) & (test_item_popularity <= 5)).sum(),
    ((test_item_popularity >= 6) & (test_item_popularity <= 10)).sum(),
    ((test_item_popularity >= 11) & (test_item_popularity <= 50)).sum(),
    (test_item_popularity > 50).sum()
]
ax6.bar(categories, counts, alpha=0.7, color='coral')
ax6.set_ylabel('Test 아이템 수')
ax6.set_title('Test 아이템의 Train 등장 횟수 분포')
ax6.grid(True, alpha=0.3, axis='y')

# 8-7. User overlap ratio distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(overlap_ratios, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
ax7.set_xlabel('Test 아이템이 Train에 등장한 비율')
ax7.set_ylabel('User 수')
ax7.set_title('User별 Test-Train 아이템 Overlap')
ax7.grid(True, alpha=0.3)

# 8-8. Train vs Test 아이템 집합
ax8 = fig.add_subplot(gs[2, 1])
from matplotlib_venn import venn2
venn2([train_items, test_items], set_labels=('Train Items', 'Test Items'), ax=ax8)
ax8.set_title('Train vs Test 아이템 Overlap')

# 8-9. Sparsity 비교
ax9 = fig.add_subplot(gs[2, 2])
sparsity_data = [train_sparsity * 100, test_sparsity * 100]
bars = ax9.bar(['Train', 'Test'], sparsity_data, alpha=0.7, color=['skyblue', 'salmon'])
ax9.set_ylabel('Sparsity (%)')
ax9.set_title('Train vs Test Sparsity')
ax9.set_ylim([99.0, 100.0])
ax9.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, sparsity_data):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}%', ha='center', va='bottom')

plt.savefig('../results/breakthrough_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 시각화 저장: results/breakthrough_analysis.png")

# 9. 핵심 인사이트 요약
print("\n" + "=" * 80)
print("핵심 인사이트 & BREAKTHROUGH 방향")
print("=" * 80)

print("\n🚨 문제점:")
print(f"1. Cold-start 심각: Test 샘플의 {len(test_cold_samples)/len(test_df)*100:.1f}%가 cold-start 아이템")
print(f"2. Long-tail 심각: Test 아이템의 {(test_item_popularity <= 10).sum()/len(test_item_popularity)*100:.1f}%가 Train에서 ≤10번 등장")
print(f"3. Extreme sparsity: {train_sparsity*100:.4f}% → GNN message passing 제한적")
print(f"4. Test가 더 어려움: 평균 rating {test_df['rating'].mean():.3f} vs {train_df['rating'].mean():.3f}")

print("\n💡 BREAKTHROUGH 아이디어:")
print("\n[A] Data-centric Approaches:")
print("  1. User/Item Bias 추가: 개인별/아이템별 선호도 편향 모델링")
print("  2. Rating을 continuous로 활용: Threshold 대신 regression")
print("  3. User normalization: 평점 스케일 차이 보정")
print("  4. Train 데이터 augmentation: Popular item도 학습 강화")

print("\n[B] Model Architecture:")
print("  5. Attention mechanism: 중요한 neighbor 강조 (GAT)")
print("  6. Higher-order connectivity: 3-4 layer로 증가")
print("  7. Residual connections: Layer 간 정보 보존")
print("  8. Node feature enrichment: Degree, popularity 등 추가 feature")

print("\n[C] Training Strategy:")
print("  9. Hard negative sampling: Low-rating item을 negative로 사용")
print(" 10. Curriculum learning: Easy → Hard 순서로 학습")
print(" 11. Multi-task learning: Rating regression + ranking 동시 학습")
print(" 12. Contrastive learning: Self-supervised pretraining")

print("\n[D] Inference Strategy:")
print(" 13. Ensemble: LightGCN + MF + Popularity 조합")
print(" 14. Re-ranking: Diversity, coverage 고려")
print(" 15. Calibration: Score calibration으로 threshold 최적화")

print("\n[E] Hybrid Approaches:")
print(" 16. Content-based features: 만약 메타데이터 있으면 활용")
print(" 17. User/Item clustering: Community detection")
print(" 18. Transfer learning: Pretrained embedding 활용")

print("\n🎯 즉시 시도할 TOP 3 아이디어:")
print("  ⭐ #1: User/Item Bias 추가 - 개인별 선호도 편향 명시적 모델링")
print("  ⭐ #2: Rating Regression + Ranking 멀티태스크 - Rating 정보 직접 활용")
print("  ⭐ #3: Hard Negative Sampling - Low-rating을 negative로 사용")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
