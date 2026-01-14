"""
Comparative Analysis: Baseline vs Proposed Method
Generates final comparison tables and publication-ready figures
for multi-document summarization research paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# STEP 1: LOAD DATA AND MERGE
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOADING AND MERGING EVALUATION RESULTS")
print("="*70)

# Load baseline results
print("Loading baseline_evaluation_results.csv...")
baseline_df = pd.read_csv('data/processed/baseline_evaluation_results.csv')
print(f"  Loaded {len(baseline_df)} clusters")

# Load proposed results
print("Loading proposed_evaluation_results.csv...")
proposed_df = pd.read_csv('data/processed/proposed_evaluation_results.csv')
print(f"  Loaded {len(proposed_df)} clusters")

# Merge on event_cluster_id
print("Merging on event_cluster_id...")
merged_df = pd.merge(
    baseline_df,
    proposed_df,
    on='event_cluster_id',
    suffixes=('_baseline', '_proposed'),
    how='inner'
)

print(f"  Merged: {len(merged_df)} clusters (matching clusters)")
print(f"  Columns: {len(merged_df.columns)}")

# Verify same clusters
print(f"\nDataset verification:")
print(f"  Baseline clusters: {len(baseline_df)}")
print(f"  Proposed clusters: {len(proposed_df)}")
print(f"  Merged clusters: {len(merged_df)}")

if len(merged_df) != len(baseline_df) or len(merged_df) != len(proposed_df):
    print(f"  WARNING: Cluster mismatch!")
else:
    print(f"  Status: All clusters matched successfully")

# ============================================================
# STEP 2: COMPUTE IMPROVEMENTS
# ============================================================
print("\n" + "="*70)
print("STEP 2: COMPUTING IMPROVEMENTS")
print("="*70)

metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore_F1']

print("Computing absolute and percentage improvements...\n")

for metric in metrics:
    col_baseline = f'{metric}_baseline'
    col_proposed = f'{metric}_proposed'
    col_abs_imp = f'{metric}_abs_improvement'
    col_pct_imp = f'{metric}_pct_improvement'
    
    # Absolute improvement
    merged_df[col_abs_imp] = merged_df[col_proposed] - merged_df[col_baseline]
    
    # Percentage improvement (avoid division by zero)
    merged_df[col_pct_imp] = np.where(
        merged_df[col_baseline] != 0,
        (merged_df[col_proposed] - merged_df[col_baseline]) / merged_df[col_baseline] * 100,
        0
    )
    
    # Compute aggregates
    mean_baseline = merged_df[col_baseline].mean()
    mean_proposed = merged_df[col_proposed].mean()
    mean_abs_imp = merged_df[col_abs_imp].mean()
    mean_pct_imp = merged_df[col_pct_imp].mean()
    
    std_abs_imp = merged_df[col_abs_imp].std()
    
    print(f"{metric}:")
    print(f"  Baseline:      {mean_baseline:.4f}")
    print(f"  Proposed:      {mean_proposed:.4f}")
    print(f"  Abs Change:    {mean_abs_imp:+.4f} +/- {std_abs_imp:.4f}")
    print(f"  Pct Change:    {mean_pct_imp:+.2f}%")
    print()

# ============================================================
# STEP 3: AGGREGATE RESULTS & CREATE SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("STEP 3: CREATING SUMMARY COMPARISON TABLE")
print("="*70)

summary_data = []

for metric in metrics:
    col_baseline = f'{metric}_baseline'
    col_proposed = f'{metric}_proposed'
    col_abs_imp = f'{metric}_abs_improvement'
    col_pct_imp = f'{metric}_pct_improvement'
    
    mean_baseline = merged_df[col_baseline].mean()
    std_baseline = merged_df[col_baseline].std()
    
    mean_proposed = merged_df[col_proposed].mean()
    std_proposed = merged_df[col_proposed].std()
    
    mean_abs_imp = merged_df[col_abs_imp].mean()
    std_abs_imp = merged_df[col_abs_imp].std()
    
    mean_pct_imp = merged_df[col_pct_imp].mean()
    
    summary_data.append({
        'Metric': metric,
        'Baseline_Mean': mean_baseline,
        'Baseline_Std': std_baseline,
        'Proposed_Mean': mean_proposed,
        'Proposed_Std': std_proposed,
        'Abs_Improvement': mean_abs_imp,
        'Abs_Improvement_Std': std_abs_imp,
        'Pct_Improvement': mean_pct_imp
    })

summary_df = pd.DataFrame(summary_data)

print("\nComparison Summary Table:")
print(summary_df.to_string(index=False))

# Save summary table
summary_file = 'data/processed/comparison_summary_table.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary table saved: {summary_file}")

# ============================================================
# STEP 4: CATEGORY-WISE COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 4: CATEGORY-WISE COMPARISON")
print("="*70)

# Group by category
category_comparison = []

for category in sorted(merged_df['news_category_baseline'].unique()):
    cat_df = merged_df[merged_df['news_category_baseline'] == category]
    
    baseline_rouge_l = cat_df['ROUGE-L_baseline'].mean()
    proposed_rouge_l = cat_df['ROUGE-L_proposed'].mean()
    rouge_l_imp = proposed_rouge_l - baseline_rouge_l
    rouge_l_pct = (proposed_rouge_l - baseline_rouge_l) / baseline_rouge_l * 100 if baseline_rouge_l != 0 else 0
    
    baseline_bert = cat_df['BERTScore_F1_baseline'].mean()
    proposed_bert = cat_df['BERTScore_F1_proposed'].mean()
    bert_imp = proposed_bert - baseline_bert
    bert_pct = (proposed_bert - baseline_bert) / baseline_bert * 100 if baseline_bert != 0 else 0
    
    n_clusters = len(cat_df)
    
    category_comparison.append({
        'news_category': category,
        'num_clusters': n_clusters,
        'ROUGE-L_Baseline': baseline_rouge_l,
        'ROUGE-L_Proposed': proposed_rouge_l,
        'ROUGE-L_Improvement': rouge_l_imp,
        'ROUGE-L_Pct_Improvement': rouge_l_pct,
        'BERTScore_F1_Baseline': baseline_bert,
        'BERTScore_F1_Proposed': proposed_bert,
        'BERTScore_F1_Improvement': bert_imp,
        'BERTScore_F1_Pct_Improvement': bert_pct
    })

category_df = pd.DataFrame(category_comparison)

print("\nCategory-wise Comparison:")
print(category_df[['news_category', 'num_clusters', 'ROUGE-L_Baseline', 'ROUGE-L_Proposed', 'ROUGE-L_Improvement']].to_string(index=False))

# Save category comparison
category_file = 'data/processed/categorywise_comparison.csv'
category_df.to_csv(category_file, index=False)
print(f"\nCategory comparison saved: {category_file}")

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
print("\n" + "="*70)
print("STEP 5: GENERATING COMPARISON VISUALIZATIONS")
print("="*70)

# Figure 1: ROUGE Comparison (Baseline vs Proposed)
print("Creating Figure 1: ROUGE Comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

metrics_rouge = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
x = np.arange(len(metrics_rouge))
width = 0.35

baseline_values = [
    merged_df['ROUGE-1_baseline'].mean(),
    merged_df['ROUGE-2_baseline'].mean(),
    merged_df['ROUGE-L_baseline'].mean()
]

proposed_values = [
    merged_df['ROUGE-1_proposed'].mean(),
    merged_df['ROUGE-2_proposed'].mean(),
    merged_df['ROUGE-L_proposed'].mean()
]

baseline_stds = [
    merged_df['ROUGE-1_baseline'].std(),
    merged_df['ROUGE-2_baseline'].std(),
    merged_df['ROUGE-L_baseline'].std()
]

proposed_stds = [
    merged_df['ROUGE-1_proposed'].std(),
    merged_df['ROUGE-2_proposed'].std(),
    merged_df['ROUGE-L_proposed'].std()
]

bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (facebook/bart-large-cnn)',
               color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, proposed_values, width, label='Proposed (Importance-Aware)',
               color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1.5)

# Add error bars
ax.errorbar(x - width/2, baseline_values, yerr=baseline_stds, fmt='none',
            color='black', capsize=5, capthick=2, alpha=0.7)
ax.errorbar(x + width/2, proposed_values, yerr=proposed_stds, fmt='none',
            color='black', capsize=5, capthick=2, alpha=0.7)

# Add value labels
for bars, values in [(bars1, baseline_values), (bars2, proposed_values)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_xlabel('ROUGE Metric', fontsize=12, fontweight='bold')
ax.set_title('Method Comparison: ROUGE Scores\n(Baseline vs Proposed Importance-Aware Method)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_rouge)
ax.set_ylim([0, 0.8])
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file1 = 'data/processed/comparison_rouge.png'
plt.savefig(fig_file1, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {fig_file1}")
plt.close()

# Figure 2: BERTScore F1 Comparison
print("Creating Figure 2: BERTScore F1 Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

baseline_bert = merged_df['BERTScore_F1_baseline'].mean()
proposed_bert = merged_df['BERTScore_F1_proposed'].mean()
baseline_bert_std = merged_df['BERTScore_F1_baseline'].std()
proposed_bert_std = merged_df['BERTScore_F1_proposed'].std()

methods = ['Baseline\n(facebook/bart-large-cnn)', 'Proposed\n(Importance-Aware)']
values = [baseline_bert, proposed_bert]
stds = [baseline_bert_std, proposed_bert_std]
colors = ['#2E86AB', '#F18F01']

bars = ax.bar(methods, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
ax.errorbar(methods, values, yerr=stds, fmt='none', color='black', capsize=8, capthick=2, alpha=0.8)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
ax.set_title('Method Comparison: BERTScore F1 Scores\n(Baseline vs Proposed Importance-Aware Method)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file2 = 'data/processed/comparison_bertscore.png'
plt.savefig(fig_file2, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {fig_file2}")
plt.close()

# Figure 3: Category-wise ROUGE-L Comparison
print("Creating Figure 3: Category-wise ROUGE-L Comparison...")
fig, ax = plt.subplots(figsize=(14, 8))

categories = category_df['news_category'].values
baseline_rouge = category_df['ROUGE-L_Baseline'].values
proposed_rouge = category_df['ROUGE-L_Proposed'].values

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_rouge, width, label='Baseline (facebook/bart-large-cnn)',
               color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, proposed_rouge, width, label='Proposed (Importance-Aware)',
               color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('ROUGE-L F1 Score', fontsize=12, fontweight='bold')
ax.set_xlabel('News Category', fontsize=12, fontweight='bold')
ax.set_title('Method Comparison: ROUGE-L by News Category\n(Baseline vs Proposed Importance-Aware Method)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_ylim([0, 0.6])
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_file3 = 'data/processed/comparison_categorywise_rouge.png'
plt.savefig(fig_file3, dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {fig_file3}")
plt.close()

# ============================================================
# STEP 6: DIAGNOSTICS
# ============================================================
print("\n" + "="*70)
print("STEP 6: COMPARISON DIAGNOSTICS")
print("="*70)

print(f"\nClusters Compared: {len(merged_df)}")

print(f"\nMean Improvements (Proposed vs Baseline):")
print(f"  ROUGE-1 (absolute): {merged_df['ROUGE-1_abs_improvement'].mean():+.4f}")
print(f"  ROUGE-1 (percent):  {merged_df['ROUGE-1_pct_improvement'].mean():+.2f}%")
print(f"  ROUGE-2 (absolute): {merged_df['ROUGE-2_abs_improvement'].mean():+.4f}")
print(f"  ROUGE-2 (percent):  {merged_df['ROUGE-2_pct_improvement'].mean():+.2f}%")
print(f"  ROUGE-L (absolute): {merged_df['ROUGE-L_abs_improvement'].mean():+.4f}")
print(f"  ROUGE-L (percent):  {merged_df['ROUGE-L_pct_improvement'].mean():+.2f}%")
print(f"  BERTScore_F1 (absolute): {merged_df['BERTScore_F1_abs_improvement'].mean():+.4f}")
print(f"  BERTScore_F1 (percent):  {merged_df['BERTScore_F1_pct_improvement'].mean():+.2f}%")

# Compute improvement statistics
print(f"\nImprovement Statistics:")
print(f"  Clusters with ROUGE-L improvement: {(merged_df['ROUGE-L_abs_improvement'] > 0).sum()} / {len(merged_df)}")
print(f"  Clusters with BERTScore_F1 improvement: {(merged_df['BERTScore_F1_abs_improvement'] > 0).sum()} / {len(merged_df)}")

print(f"\nCategory Distribution:")
for _, row in category_df.iterrows():
    print(f"  {row['news_category']}: {row['num_clusters']} clusters")

# ============================================================
# EXECUTION COMPLETE
# ============================================================
print("\n" + "="*70)
print("EXECUTION COMPLETE")
print("="*70)

print(f"\nOutput Files Generated:")
print(f"  1. {summary_file}")
print(f"  2. {category_file}")
print(f"  3. {fig_file1}")
print(f"  4. {fig_file2}")
print(f"  5. {fig_file3}")

print(f"\nKey Findings:")
print(f"  - {len(merged_df)} multi-document clusters compared")
print(f"  - Baseline: facebook/bart-large-cnn (no importance weighting)")
print(f"  - Proposed: Importance-aware ordering (h_i, alpha_i, w_i formulation)")
print(f"  - Fair comparison: Identical evaluation methodology")
print(f"  - All figures are publication-ready (300 DPI)")

print("\nReady for Research Paper:")
print(f"  - Summary comparison table (CSV)")
print(f"  - Category-wise breakdown (CSV)")
print(f"  - Publication-quality figures (PNG, 300 DPI)")
print(f"  - Statistical improvements documented")
