"""
Exploratory Data Analysis for GNN-based Recommendation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Load data
print("="*80)
print("GNN-RecSys: Exploratory Data Analysis")
print("="*80)

data = pd.read_csv('../data/train.csv')
print(f"\n[1] Dataset Overview")
print(f"{'='*80}")
print(f"Total interactions: {len(data):,}")
print(f"Columns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head(10))
print(f"\nData types:")
print(data.dtypes)
print(f"\nMissing values:")
print(data.isnull().sum())

# Basic statistics
print(f"\n[2] Basic Statistics")
print(f"{'='*80}")
n_users = data['user'].nunique()
n_items = data['item'].nunique()
n_interactions = len(data)

print(f"Unique users: {n_users:,}")
print(f"Unique items: {n_items:,}")
print(f"Total interactions: {n_interactions:,}")
print(f"\nAverage interactions per user: {n_interactions/n_users:.2f}")
print(f"Average interactions per item: {n_interactions/n_items:.2f}")
print(f"\nData sparsity: {100 * (1 - n_interactions / (n_users * n_items)):.4f}%")
print(f"Data density: {100 * n_interactions / (n_users * n_items):.4f}%")

# Rating statistics
print(f"\n[3] Rating Distribution")
print(f"{'='*80}")
print(data['rating'].describe())
print(f"\nRating value counts:")
rating_counts = data['rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    percentage = 100 * count / len(data)
    print(f"Rating {rating}: {count:,} ({percentage:.2f}%)")

# User analysis
print(f"\n[4] User Interaction Analysis")
print(f"{'='*80}")
user_interactions = data.groupby('user').size()
print(f"Min interactions per user: {user_interactions.min()}")
print(f"Max interactions per user: {user_interactions.max()}")
print(f"Mean interactions per user: {user_interactions.mean():.2f}")
print(f"Median interactions per user: {user_interactions.median():.2f}")
print(f"Std interactions per user: {user_interactions.std():.2f}")

print(f"\nUser interaction percentiles:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(user_interactions, p)
    print(f"  {p}th percentile: {val:.0f} interactions")

# Top active users
print(f"\nTop 10 most active users:")
top_users = user_interactions.nlargest(10)
for user_id, count in top_users.items():
    print(f"  User {user_id}: {count} interactions")

# Item analysis
print(f"\n[5] Item (Movie) Popularity Analysis")
print(f"{'='*80}")
item_popularity = data.groupby('item').size()
print(f"Min interactions per item: {item_popularity.min()}")
print(f"Max interactions per item: {item_popularity.max()}")
print(f"Mean interactions per item: {item_popularity.mean():.2f}")
print(f"Median interactions per item: {item_popularity.median():.2f}")
print(f"Std interactions per item: {item_popularity.std():.2f}")

print(f"\nItem popularity percentiles:")
for p in percentiles:
    val = np.percentile(item_popularity, p)
    print(f"  {p}th percentile: {val:.0f} interactions")

# Long tail analysis
items_with_1_interaction = (item_popularity == 1).sum()
items_with_le_5 = (item_popularity <= 5).sum()
items_with_le_10 = (item_popularity <= 10).sum()
print(f"\nLong-tail characteristics:")
print(f"  Items with only 1 interaction: {items_with_1_interaction} ({100*items_with_1_interaction/n_items:.2f}%)")
print(f"  Items with ≤5 interactions: {items_with_le_5} ({100*items_with_le_5/n_items:.2f}%)")
print(f"  Items with ≤10 interactions: {items_with_le_10} ({100*items_with_le_10/n_items:.2f}%)")

print(f"\nTop 20 most popular items:")
top_items = item_popularity.nlargest(20)
for item_id, count in top_items.items():
    avg_rating = data[data['item'] == item_id]['rating'].mean()
    print(f"  Item {item_id}: {count} interactions (avg rating: {avg_rating:.2f})")

# Rating patterns by user
print(f"\n[6] Rating Patterns")
print(f"{'='*80}")
user_avg_ratings = data.groupby('user')['rating'].mean()
print(f"Average rating per user:")
print(f"  Min: {user_avg_ratings.min():.2f}")
print(f"  Max: {user_avg_ratings.max():.2f}")
print(f"  Mean: {user_avg_ratings.mean():.2f}")
print(f"  Median: {user_avg_ratings.median():.2f}")
print(f"  Std: {user_avg_ratings.std():.2f}")

# High vs low raters
high_raters = (user_avg_ratings >= 4.0).sum()
medium_raters = ((user_avg_ratings >= 3.0) & (user_avg_ratings < 4.0)).sum()
low_raters = (user_avg_ratings < 3.0).sum()
print(f"\nUser rating tendencies:")
print(f"  High raters (avg ≥4.0): {high_raters} ({100*high_raters/n_users:.2f}%)")
print(f"  Medium raters (3.0≤avg<4.0): {medium_raters} ({100*medium_raters/n_users:.2f}%)")
print(f"  Low raters (avg<3.0): {low_raters} ({100*low_raters/n_users:.2f}%)")

# Item ID analysis
print(f"\n[7] Item ID Distribution")
print(f"{'='*80}")
item_ids = data['item'].unique()
print(f"Min item ID: {item_ids.min()}")
print(f"Max item ID: {item_ids.max()}")
print(f"Item ID range: {item_ids.max() - item_ids.min() + 1}")
print(f"Actual unique items: {len(item_ids)}")
print(f"ID gaps: {item_ids.max() - item_ids.min() + 1 - len(item_ids)}")

# Check if IDs are continuous
sorted_items = sorted(item_ids)
gaps = []
for i in range(len(sorted_items)-1):
    gap = sorted_items[i+1] - sorted_items[i]
    if gap > 1:
        gaps.append((sorted_items[i], sorted_items[i+1], gap-1))

print(f"\nTotal ID gaps: {len(gaps)}")
if len(gaps) > 0 and len(gaps) <= 10:
    print("Gap details:")
    for start, end, gap_size in gaps[:10]:
        print(f"  Between {start} and {end}: {gap_size} missing IDs")

# Positive/Negative threshold analysis
print(f"\n[8] Threshold Analysis for Binary Classification")
print(f"{'='*80}")
thresholds = [2.5, 3.0, 3.5, 4.0, 4.5]
for threshold in thresholds:
    positive = (data['rating'] >= threshold).sum()
    negative = (data['rating'] < threshold).sum()
    pos_ratio = 100 * positive / len(data)
    print(f"Threshold ≥{threshold}: {positive:,} positive ({pos_ratio:.2f}%), {negative:,} negative ({100-pos_ratio:.2f}%)")

# Graph structure insights
print(f"\n[9] Graph Structure Insights")
print(f"{'='*80}")
print(f"Bipartite graph: {n_users} user nodes + {n_items} item nodes = {n_users + n_items} total nodes")
print(f"Edges: {n_interactions:,}")
print(f"Average degree (user side): {n_interactions / n_users:.2f}")
print(f"Average degree (item side): {n_interactions / n_items:.2f}")

# Potential issues for GNN
print(f"\n[10] Potential Challenges for GNN")
print(f"{'='*80}")
print(f"✓ Very sparse graph ({100 * (1 - n_interactions / (n_users * n_items)):.4f}% sparsity)")
print(f"✓ Highly skewed item popularity (long tail distribution)")
print(f"✓ User activity varies widely ({user_interactions.min()} to {user_interactions.max()} interactions)")
print(f"✓ Non-continuous item IDs (need re-indexing)")
print(f"✓ Imbalanced ratings (skewed towards positive)")

# Summary statistics
print(f"\n[11] Summary & Recommendations")
print(f"{'='*80}")
print("Key Findings:")
print(f"  1. Dataset size: {n_users} users × {n_items} items = {n_interactions:,} interactions")
print(f"  2. Sparsity: {100 * (1 - n_interactions / (n_users * n_items)):.4f}%")
print(f"  3. Average user activity: {n_interactions/n_users:.1f} interactions")
print(f"  4. Rating bias: Mostly positive (4.0 is most common)")
print(f"  5. Long-tail: {100*items_with_le_10/n_items:.1f}% of items have ≤10 interactions")

print("\nRecommendations:")
print("  • Use threshold 3.5 or 4.0 for positive labels (balanced split)")
print("  • Re-index item IDs to be continuous (0 to N-1)")
print("  • Consider user/item cold-start handling")
print("  • Implement negative sampling strategy")
print("  • Use graph pooling or attention to handle varying node degrees")

print(f"\n{'='*80}")
print("Analysis complete! Generating visualizations...")
print(f"{'='*80}\n")
