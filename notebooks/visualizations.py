"""
Visualization for EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('../data/train.csv')

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Rating Distribution
ax1 = plt.subplot(3, 3, 1)
rating_counts = data['rating'].value_counts().sort_index()
ax1.bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Rating')
ax1.set_ylabel('Count')
ax1.set_title('Rating Distribution')
ax1.grid(axis='y', alpha=0.3)
for i, (rating, count) in enumerate(rating_counts.items()):
    ax1.text(rating, count, f'{count:,}', ha='center', va='bottom', fontsize=8)

# 2. User Interaction Distribution
ax2 = plt.subplot(3, 3, 2)
user_interactions = data.groupby('user').size()
ax2.hist(user_interactions, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Number of Interactions')
ax2.set_ylabel('Number of Users')
ax2.set_title(f'User Activity Distribution (Mean: {user_interactions.mean():.1f})')
ax2.axvline(user_interactions.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(user_interactions.median(), color='green', linestyle='--', linewidth=2, label='Median')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Item Popularity Distribution (Log scale)
ax3 = plt.subplot(3, 3, 3)
item_popularity = data.groupby('item').size().sort_values(ascending=False)
ax3.plot(range(len(item_popularity)), item_popularity.values, color='purple', linewidth=2)
ax3.set_xlabel('Item Rank')
ax3.set_ylabel('Number of Interactions')
ax3.set_title('Item Popularity (Long-tail Distribution)')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# 4. Top 20 Most Popular Items
ax4 = plt.subplot(3, 3, 4)
top_20_items = item_popularity.head(20)
ax4.barh(range(len(top_20_items)), top_20_items.values, color='teal', alpha=0.7)
ax4.set_yticks(range(len(top_20_items)))
ax4.set_yticklabels([f'Item {idx}' for idx in top_20_items.index], fontsize=8)
ax4.set_xlabel('Number of Interactions')
ax4.set_title('Top 20 Most Popular Items')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# 5. Average Rating per User
ax5 = plt.subplot(3, 3, 5)
user_avg_ratings = data.groupby('user')['rating'].mean()
ax5.hist(user_avg_ratings, bins=30, color='gold', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Average Rating')
ax5.set_ylabel('Number of Users')
ax5.set_title(f'User Average Rating Distribution (Mean: {user_avg_ratings.mean():.2f})')
ax5.axvline(user_avg_ratings.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Threshold Analysis
ax6 = plt.subplot(3, 3, 6)
thresholds = [2.5, 3.0, 3.5, 4.0, 4.5]
positive_ratios = []
negative_ratios = []
for threshold in thresholds:
    positive = (data['rating'] >= threshold).sum()
    negative = (data['rating'] < threshold).sum()
    positive_ratios.append(100 * positive / len(data))
    negative_ratios.append(100 * negative / len(data))

x = np.arange(len(thresholds))
width = 0.35
ax6.bar(x - width/2, positive_ratios, width, label='Positive', color='green', alpha=0.7)
ax6.bar(x + width/2, negative_ratios, width, label='Negative', color='red', alpha=0.7)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Percentage (%)')
ax6.set_title('Positive/Negative Split by Threshold')
ax6.set_xticks(x)
ax6.set_xticklabels([f'≥{t}' for t in thresholds])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Items by Interaction Count (Long tail)
ax7 = plt.subplot(3, 3, 7)
bucket_ranges = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
bucket_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '100+']
bucket_counts = []
for lower, upper in bucket_ranges:
    if upper == float('inf'):
        count = (item_popularity >= lower).sum()
    elif lower == upper:
        count = (item_popularity == lower).sum()
    else:
        count = ((item_popularity >= lower) & (item_popularity <= upper)).sum()
    bucket_counts.append(count)

ax7.bar(range(len(bucket_counts)), bucket_counts, color='mediumpurple', alpha=0.7)
ax7.set_xticks(range(len(bucket_counts)))
ax7.set_xticklabels(bucket_labels, rotation=45)
ax7.set_xlabel('Number of Interactions')
ax7.set_ylabel('Number of Items')
ax7.set_title('Item Distribution by Interaction Count')
ax7.grid(axis='y', alpha=0.3)
for i, count in enumerate(bucket_counts):
    ax7.text(i, count, f'{count}', ha='center', va='bottom', fontsize=8)

# 8. User Interaction Percentiles
ax8 = plt.subplot(3, 3, 8)
percentiles = [10, 25, 50, 75, 90, 95, 99]
user_percentiles = [np.percentile(user_interactions, p) for p in percentiles]
ax8.plot(percentiles, user_percentiles, marker='o', markersize=8, linewidth=2, color='navy')
ax8.set_xlabel('Percentile')
ax8.set_ylabel('Number of Interactions')
ax8.set_title('User Interaction Percentiles')
ax8.grid(True, alpha=0.3)
for i, (p, val) in enumerate(zip(percentiles, user_percentiles)):
    ax8.text(p, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

# 9. Sparsity Visualization
ax9 = plt.subplot(3, 3, 9)
n_users = data['user'].nunique()
n_items = data['item'].nunique()
n_interactions = len(data)
density = 100 * n_interactions / (n_users * n_items)
sparsity = 100 - density

values = [density, sparsity]
labels = [f'Density\n{density:.2f}%', f'Sparsity\n{sparsity:.2f}%']
colors = ['lightgreen', 'lightcoral']
ax9.pie(values, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
ax9.set_title('Graph Density vs Sparsity')

plt.tight_layout()
plt.savefig('../results/eda_visualizations.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to ../results/eda_visualizations.png")

# Additional detailed plots
fig2 = plt.figure(figsize=(20, 8))

# Item popularity with average rating
ax1 = plt.subplot(1, 3, 1)
top_50_items = item_popularity.head(50)
avg_ratings = [data[data['item'] == item]['rating'].mean() for item in top_50_items.index]
scatter = ax1.scatter(top_50_items.values, avg_ratings,
                     s=100, c=avg_ratings, cmap='RdYlGn', alpha=0.6, edgecolors='black')
ax1.set_xlabel('Number of Interactions')
ax1.set_ylabel('Average Rating')
ax1.set_title('Top 50 Items: Popularity vs Quality')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Avg Rating')

# Rating distribution over time (by user ID as proxy)
ax2 = plt.subplot(1, 3, 2)
user_ids = sorted(data['user'].unique())
avg_ratings_by_user = [data[data['user'] == uid]['rating'].mean() for uid in user_ids]
ax2.plot(user_ids, avg_ratings_by_user, linewidth=1, alpha=0.6, color='darkblue')
ax2.set_xlabel('User ID')
ax2.set_ylabel('Average Rating')
ax2.set_title('Average Rating by User')
ax2.axhline(data['rating'].mean(), color='red', linestyle='--', linewidth=2, label='Overall Mean')
ax2.legend()
ax2.grid(True, alpha=0.3)

# User vs Item interaction heatmap (sampled)
ax3 = plt.subplot(1, 3, 3)
# Sample top 30 users and top 30 items for visualization
top_30_users = user_interactions.nlargest(30).index
top_30_items_idx = item_popularity.head(30).index
sample_data = data[data['user'].isin(top_30_users) & data['item'].isin(top_30_items_idx)]
pivot = sample_data.pivot_table(values='rating', index='user', columns='item', fill_value=0)
sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Rating'}, ax=ax3,
            linewidths=0.5, linecolor='gray', square=True)
ax3.set_title('Interaction Heatmap (Top 30 Users × Top 30 Items)')
ax3.set_xlabel('Item ID')
ax3.set_ylabel('User ID')

plt.tight_layout()
plt.savefig('../results/eda_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Detailed analysis saved to ../results/eda_detailed_analysis.png")

print("\nVisualization generation complete!")
