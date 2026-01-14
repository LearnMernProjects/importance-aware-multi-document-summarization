"""
Baseline Multi-Document Summarization Evaluation
Evaluates facebook/bart-large-cnn baseline model using ROUGE and BERTScore
Produces research-ready metrics and visualizations
"""

import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# STEP 1: LOADING DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

# Load baseline summaries
print("Loading baseline_summaries.csv...")
df = pd.read_csv('data/processed/baseline_summaries.csv')

initial_count = len(df)
print(f"  Loaded {initial_count} rows")

# Remove rows with missing or empty summaries
df = df.dropna(subset=['baseline_generated_summary', 'reference_summary'])
df = df[df['baseline_generated_summary'].str.strip() != '']
df = df[df['reference_summary'].str.strip() != '']

print(f"  After cleaning: {len(df)} rows")
print(f"  Removed {initial_count - len(df)} rows with missing summaries")
print(f"  Dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2: ROUGE EVALUATION
# ============================================================
print("\n" + "="*70)
print("STEP 2: ROUGE EVALUATION (ROUGE-1, ROUGE-2, ROUGE-L)")
print("="*70)

print("Initializing ROUGE scorer...")
rouge_types = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

# Store ROUGE scores
rouge_scores = {
    'ROUGE-1': [],
    'ROUGE-2': [],
    'ROUGE-L': []
}

print("Computing ROUGE scores per cluster...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE"):
    generated = row['baseline_generated_summary']
    reference = row['reference_summary']
    
    scores = scorer.score(reference, generated)
    
    rouge_scores['ROUGE-1'].append(scores['rouge1'].fmeasure)
    rouge_scores['ROUGE-2'].append(scores['rouge2'].fmeasure)
    rouge_scores['ROUGE-L'].append(scores['rougeL'].fmeasure)

# Add to dataframe
df['ROUGE-1'] = rouge_scores['ROUGE-1']
df['ROUGE-2'] = rouge_scores['ROUGE-2']
df['ROUGE-L'] = rouge_scores['ROUGE-L']

# Compute averages
avg_rouge1 = np.mean(rouge_scores['ROUGE-1'])
avg_rouge2 = np.mean(rouge_scores['ROUGE-2'])
avg_rougel = np.mean(rouge_scores['ROUGE-L'])

print(f"✓ ROUGE scores computed")
print(f"  Average ROUGE-1: {avg_rouge1:.4f}")
print(f"  Average ROUGE-2: {avg_rouge2:.4f}")
print(f"  Average ROUGE-L: {avg_rougel:.4f}")

# ============================================================
# STEP 3: BERTSCORE EVALUATION
# ============================================================
print("\n" + "="*70)
print("STEP 3: BERTSCORE EVALUATION")
print("="*70)

print("Computing BERTScore (Precision, Recall, F1)...")
print("  (This may take a few minutes on CPU)")

generated_summaries = df['baseline_generated_summary'].tolist()
reference_summaries = df['reference_summary'].tolist()

# Compute BERTScore
P, R, F1 = bert_score(
    generated_summaries,
    reference_summaries,
    lang='en',
    model_type='microsoft/deberta-xlarge-mnli',
    device='cpu',
    batch_size=16
)

df['BERTScore_P'] = P.cpu().numpy()
df['BERTScore_R'] = R.cpu().numpy()
df['BERTScore_F1'] = F1.cpu().numpy()

avg_bertscore_f1 = np.mean(F1.cpu().numpy())
avg_bertscore_p = np.mean(P.cpu().numpy())
avg_bertscore_r = np.mean(R.cpu().numpy())

print(f"✓ BERTScore computed")
print(f"  Average BERTScore Precision: {avg_bertscore_p:.4f}")
print(f"  Average BERTScore Recall: {avg_bertscore_r:.4f}")
print(f"  Average BERTScore F1: {avg_bertscore_f1:.4f}")

# ============================================================
# STEP 4: CATEGORY-WISE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 4: CATEGORY-WISE ANALYSIS")
print("="*70)

# Aggregate by category
category_analysis = df.groupby('news_category').agg({
    'ROUGE-1': ['mean', 'std', 'count'],
    'ROUGE-2': ['mean', 'std'],
    'ROUGE-L': ['mean', 'std'],
    'BERTScore_F1': ['mean', 'std'],
    'num_articles_in_cluster': 'mean'
}).round(4)

print("\nCategory-wise Performance:")
print(category_analysis)

# Prepare for visualization
category_rouge_l = df.groupby('news_category')['ROUGE-L'].mean().sort_values(ascending=False)
category_bert_f1 = df.groupby('news_category')['BERTScore_F1'].mean().sort_values(ascending=False)

print("\nTop 5 categories by ROUGE-L:")
print(category_rouge_l.head())

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("STEP 5: SAVING EVALUATION RESULTS")
print("="*70)

# Select columns for output
output_cols = [
    'event_cluster_id',
    'ROUGE-1',
    'ROUGE-2',
    'ROUGE-L',
    'BERTScore_P',
    'BERTScore_R',
    'BERTScore_F1',
    'news_category',
    'num_articles_in_cluster'
]

df_output = df[output_cols].copy()
output_file = 'data/processed/baseline_evaluation_results.csv'
df_output.to_csv(output_file, index=False)

print(f"Evaluation results saved: {output_file}")
print(f"  Rows: {len(df_output)}")
print(f"  Columns: {list(output_cols)}")

# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================
print("\n" + "="*70)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*70)

# Figure 1: Overall ROUGE Scores
print("Creating Figure 1: Overall ROUGE Scores...")
fig, ax = plt.subplots(figsize=(10, 6))

rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
rouge_values = [avg_rouge1, avg_rouge2, avg_rougel]
rouge_stds = [
    np.std(rouge_scores['ROUGE-1']),
    np.std(rouge_scores['ROUGE-2']),
    np.std(rouge_scores['ROUGE-L'])
]

colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax.bar(rouge_metrics, rouge_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add error bars
ax.errorbar(rouge_metrics, rouge_values, yerr=rouge_stds, fmt='none', 
            color='black', capsize=5, capthick=2, label='Std Dev')

# Add value labels on bars
for bar, val in zip(bars, rouge_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_xlabel('ROUGE Metric', fontsize=12, fontweight='bold')
ax.set_title('Baseline Model: Overall ROUGE Scores\n(facebook/bart-large-cnn)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
fig_file1 = 'data/processed/baseline_rouge_scores.png'
plt.savefig(fig_file1, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {fig_file1}")
plt.close()

# Figure 2: Category-wise ROUGE-L Performance
print("Creating Figure 2: Category-wise ROUGE-L Scores...")
fig, ax = plt.subplots(figsize=(12, 7))

categories = category_rouge_l.index
values = category_rouge_l.values
colors_cat = plt.cm.viridis(np.linspace(0, 1, len(categories)))

bars = ax.barh(categories, values, color=colors_cat, alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val, bar.get_y() + bar.get_height()/2.,
            f' {val:.4f}',
            va='center', ha='left', fontsize=10, fontweight='bold')

ax.set_xlabel('ROUGE-L F1 Score', fontsize=12, fontweight='bold')
ax.set_ylabel('News Category', fontsize=12, fontweight='bold')
ax.set_title('Baseline Model: ROUGE-L Performance by News Category\n(facebook/bart-large-cnn)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim([0, 1.0])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_file2 = 'data/processed/baseline_categorywise_rouge.png'
plt.savefig(fig_file2, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {fig_file2}")
plt.close()

# Figure 3: BERTScore Distribution by Category
print("Creating Figure 3: BERTScore F1 Distribution...")
fig, ax = plt.subplots(figsize=(12, 7))

# Box plot
category_order = df.groupby('news_category')['BERTScore_F1'].median().sort_values(ascending=False).index
df_sorted = df.copy()
df_sorted['news_category'] = pd.Categorical(df_sorted['news_category'], categories=category_order, ordered=True)

sns.boxplot(data=df_sorted, y='news_category', x='BERTScore_F1', 
            palette='Set2', ax=ax, width=0.6)

ax.set_xlabel('BERTScore F1', fontsize=12, fontweight='bold')
ax.set_ylabel('News Category', fontsize=12, fontweight='bold')
ax.set_title('Baseline Model: BERTScore F1 Distribution by News Category\n(facebook/bart-large-cnn)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim([0, 1.0])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_file3 = 'data/processed/baseline_bertscore_distribution.png'
plt.savefig(fig_file3, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {fig_file3}")
plt.close()

# ============================================================
# STEP 7: DIAGNOSTICS
# ============================================================
print("\n" + "="*70)
print("STEP 7: EVALUATION DIAGNOSTICS")
print("="*70)

total_clusters = len(df)
avg_articles_per_cluster = df['num_articles_in_cluster'].mean()
std_articles_per_cluster = df['num_articles_in_cluster'].std()

print(f"\nDataset Summary:")
print(f"  Total clusters evaluated: {total_clusters}")
print(f"  Average articles per cluster: {avg_articles_per_cluster:.2f} ± {std_articles_per_cluster:.2f}")
print(f"  Range: {df['num_articles_in_cluster'].min():.0f} - {df['num_articles_in_cluster'].max():.0f}")

print(f"\nOverall ROUGE Performance:")
print(f"  ROUGE-1: {avg_rouge1:.4f} ± {np.std(rouge_scores['ROUGE-1']):.4f}")
print(f"  ROUGE-2: {avg_rouge2:.4f} ± {np.std(rouge_scores['ROUGE-2']):.4f}")
print(f"  ROUGE-L: {avg_rougel:.4f} ± {np.std(rouge_scores['ROUGE-L']):.4f}")

print(f"\nOverall BERTScore Performance:")
print(f"  BERTScore Precision: {avg_bertscore_p:.4f} ± {np.std(P.cpu().numpy()):.4f}")
print(f"  BERTScore Recall: {avg_bertscore_r:.4f} ± {np.std(R.cpu().numpy()):.4f}")
print(f"  BERTScore F1: {avg_bertscore_f1:.4f} ± {np.std(F1.cpu().numpy()):.4f}")

print(f"\nCategory Distribution:")
category_counts = df['news_category'].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat}: {count} clusters ({count/total_clusters*100:.1f}%)")

# ============================================================
# EXECUTION COMPLETE
# ============================================================
print("\n" + "="*70)
print("EXECUTION COMPLETE")
print("="*70)

print(f"\nOutput Files Generated:")
print(f"  1. Evaluation Results: {output_file}")
print(f"  2. Figure 1: {fig_file1}")
print(f"  3. Figure 2: {fig_file2}")
print(f"  4. Figure 3: {fig_file3}")

print(f"\nSummary for Research:")
print(f"  - {total_clusters} multi-document event clusters evaluated")
print(f"  - Baseline model: facebook/bart-large-cnn")
print(f"  - Evaluation metrics: ROUGE-1/2/L, BERTScore F1")
print(f"  - Results are ready for journal submission")
print(f"  - All visualizations are publication-quality (300 DPI)")

print("\nNext steps:")
print(f"  - Review baseline_evaluation_results.csv for detailed per-cluster scores")
print(f"  - Use figures for research paper/report")
print(f"  - Compare with proposed methods when available")
