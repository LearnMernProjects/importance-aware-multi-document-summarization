"""
Error Analysis - Individual Image Generation
Creates separate high-quality images for each metric with clear titles and proper formatting
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'

print("=" * 100)
print("GENERATING INDIVIDUAL HIGH-QUALITY IMAGES")
print("=" * 100)

# Load data
results_df = pd.read_csv('data/processed/error_analysis.csv')
comparison_df = pd.read_csv('data/processed/error_analysis_comparison.csv')
category_df = pd.read_csv('data/processed/error_analysis_by_category.csv')

# ==================== IMAGE 1: OVERALL METRICS COMPARISON ====================
print("\n[1/6] Creating Overall Metrics Comparison...")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#ffffff')

metrics = comparison_df['Metric'].values
baseline = comparison_df['Baseline'].values
aims = comparison_df['AIMS'].values

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#FF6B6B', alpha=0.85, edgecolor='#CC0000', linewidth=2.5)
bars2 = ax.bar(x + width/2, aims, width, label='AIMS (Proposed)', color='#4ECDC4', alpha=0.85, edgecolor='#008080', linewidth=2.5)

ax.set_ylabel('Error Rate', fontsize=14, fontweight='bold')
ax.set_title('Error Metrics Comparison: Baseline vs AIMS\nLower Values are Better', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Redundancy Rate', 'Omission Rate', 'Hallucination Rate'], fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
ax.set_ylim(0, max(baseline) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add info box
info_text = (f"Dataset: {len(results_df)//2} event clusters\n"
             f"Categories: {len(results_df['category'].unique())}\n"
             f"Total Summaries: {len(results_df)}")
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F0F0', alpha=0.95, edgecolor='#333333', linewidth=1.5))

plt.tight_layout()
plt.savefig('data/processed/01_metrics_comparison.png', dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none')
print("[OK] Image saved: 01_metrics_comparison.png")
plt.close()

# ==================== IMAGE 2: REDUNDANCY RATE ====================
print("[2/6] Creating Redundancy Rate Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#ffffff')

baseline_red = comparison_df[comparison_df['Metric'] == 'Redundancy Rate']['Baseline'].values[0]
aims_red = comparison_df[comparison_df['Metric'] == 'Redundancy Rate']['AIMS'].values[0]

bars = ax.bar(['Baseline', 'AIMS (Proposed)'], [baseline_red, aims_red], 
              color=['#FF6B6B', '#4ECDC4'], alpha=0.85, edgecolor=['#CC0000', '#008080'], linewidth=3, width=0.6)

ax.set_ylabel('Redundancy Rate', fontsize=14, fontweight='bold')
ax.set_title('Redundancy Rate: Measure of Repeated Content\nLower = Better (No Repeated N-grams or Sentences)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(baseline_red, aims_red) * 1.3)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, [baseline_red, aims_red])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.6f}\n({val*100:.4f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add description
desc = ("Calculation Method:\n"
        "• Extract all 3-grams\n"
        "• Count repeated instances\n"
        "• Calculate sentence similarity (TF-IDF)\n"
        "• Combine: 60% n-gram + 40% similarity")
ax.text(0.98, 0.65, desc, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', alpha=0.95, edgecolor='#FF6F00', linewidth=1.5))

plt.tight_layout()
plt.savefig('data/processed/02_redundancy_rate.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 02_redundancy_rate.png")
plt.close()

# ==================== IMAGE 3: OMISSION RATE ====================
print("[3/6] Creating Omission Rate Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#ffffff')

baseline_omit = comparison_df[comparison_df['Metric'] == 'Omission Rate']['Baseline'].values[0]
aims_omit = comparison_df[comparison_df['Metric'] == 'Omission Rate']['AIMS'].values[0]
improvement_omit = comparison_df[comparison_df['Metric'] == 'Omission Rate']['Improvement (%)'].values[0]

bars = ax.bar(['Baseline', 'AIMS (Proposed)'], [baseline_omit, aims_omit], 
              color=['#FF6B6B', '#4ECDC4'], alpha=0.85, edgecolor=['#CC0000', '#008080'], linewidth=3, width=0.6)

ax.set_ylabel('Omission Rate', fontsize=14, fontweight='bold')
ax.set_title('Omission Rate: Named Entity Coverage\nLower = Better (Fewer Missing Entities from References)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(baseline_omit, aims_omit) * 1.2)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, [baseline_omit, aims_omit])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement box
improve_color = '#90EE90' if improvement_omit > 0 else '#FFB6C6'
improve_text = f"AIMS Improvement:\n{improvement_omit:+.2f}%\n\n✓ AIMS BETTER" if improvement_omit > 0 else f"Baseline Better\n{-improvement_omit:.2f}%"
ax.text(0.98, 0.65, improve_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=improve_color, alpha=0.95, edgecolor='#333333', linewidth=2))

# Add description
desc = ("Calculation Method:\n"
        "• Extract entities (spaCy NER)\n"
        "• Compare reference vs generated\n"
        "• Count missing entities\n"
        "• Omission% = Missing / Total Ref")
ax.text(0.02, 0.65, desc, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', alpha=0.95, edgecolor='#0088CC', linewidth=1.5))

plt.tight_layout()
plt.savefig('data/processed/03_omission_rate.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 03_omission_rate.png")
plt.close()

# ==================== IMAGE 4: HALLUCINATION RATE ====================
print("[4/6] Creating Hallucination Rate Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#ffffff')

baseline_hall = comparison_df[comparison_df['Metric'] == 'Hallucination Rate']['Baseline'].values[0]
aims_hall = comparison_df[comparison_df['Metric'] == 'Hallucination Rate']['AIMS'].values[0]
improvement_hall = comparison_df[comparison_df['Metric'] == 'Hallucination Rate']['Improvement (%)'].values[0]

bars = ax.bar(['Baseline', 'AIMS (Proposed)'], [baseline_hall, aims_hall], 
              color=['#FF6B6B', '#4ECDC4'], alpha=0.85, edgecolor=['#CC0000', '#008080'], linewidth=3, width=0.6)

ax.set_ylabel('Hallucination Rate', fontsize=14, fontweight='bold')
ax.set_title('Hallucination Rate: Out-of-Context Entities\nLower = Better (Fewer Fabricated Entities)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(baseline_hall, aims_hall) * 1.2)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, [baseline_hall, aims_hall])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement box
improve_color = '#90EE90' if improvement_hall > 0 else '#FFB6C6'
improve_text = f"AIMS Better:\n{improvement_hall:+.2f}%" if improvement_hall > 0 else f"Baseline Better:\n{-improvement_hall:.2f}%"
ax.text(0.98, 0.65, improve_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=improve_color, alpha=0.95, edgecolor='#333333', linewidth=2))

# Add description
desc = ("Calculation Method:\n"
        "• Extract entities from generated\n"
        "• Check against source/reference\n"
        "• Count non-existent entities\n"
        "• Hallucination% = Invalid / Total")
ax.text(0.02, 0.65, desc, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE6E6', alpha=0.95, edgecolor='#CC0000', linewidth=1.5))

plt.tight_layout()
plt.savefig('data/processed/04_hallucination_rate.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 04_hallucination_rate.png")
plt.close()

# ==================== IMAGE 5: IMPROVEMENT PERCENTAGE ====================
print("[5/6] Creating Improvement Percentage Chart...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#ffffff')

metrics_names = comparison_df['Metric'].values
improvements = comparison_df['Improvement (%)'].values
colors = ['#90EE90' if x > 0 else '#FFB6C6' for x in improvements]

bars = ax.barh(metrics_names, improvements, color=colors, alpha=0.85, edgecolor=['#008000' if x > 0 else '#CC0000' for x in improvements], linewidth=2.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)

ax.set_xlabel('AIMS Improvement (%)', fontsize=14, fontweight='bold')
ax.set_title('AIMS Performance Improvement vs Baseline\nPositive % = AIMS Performs Better | Negative % = Baseline Performs Better', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    x_pos = val + (2 if val > 0 else -2)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2., f'{val:+.2f}%',
            ha=ha, va='center', fontsize=12, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#90EE90', edgecolor='#008000', linewidth=2, label='AIMS Better'),
    Patch(facecolor='#FFB6C6', edgecolor='#CC0000', linewidth=2, label='Baseline Better')
]
ax.legend(handles=legend_elements, fontsize=11, loc='lower right', framealpha=0.95, edgecolor='black')

plt.tight_layout()
plt.savefig('data/processed/05_improvement_percentage.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 05_improvement_percentage.png")
plt.close()

# ==================== IMAGE 6: CATEGORY-WISE OMISSION RATE ====================
print("[6/6] Creating Category-wise Performance...")

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('#ffffff')

categories = category_df['Category'].values
baseline_omission = category_df['Baseline_Omission'].values
aims_omission = category_df['AIMS_Omission'].values

x = np.arange(len(categories))
width = 0.35

bars1 = ax.barh(x - width/2, baseline_omission, width, label='Baseline', color='#FF6B6B', alpha=0.85, edgecolor='#CC0000', linewidth=2)
bars2 = ax.barh(x + width/2, aims_omission, width, label='AIMS (Proposed)', color='#4ECDC4', alpha=0.85, edgecolor='#008080', linewidth=2)

ax.set_xlabel('Omission Rate (Lower is Better)', fontsize=13, fontweight='bold')
ax.set_title('Category-wise Omission Rate Comparison\nEntity Coverage Performance by News Category', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_yticks(x)
ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(fontsize=11, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1)
ax.set_xlim(0, max(baseline_omission.max(), aims_omission.max()) * 1.15)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        width_val = bar.get_width()
        ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{width_val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('data/processed/06_category_omission_comparison.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 06_category_omission_comparison.png")
plt.close()

# ==================== IMAGE 7: DISTRIBUTION ANALYSIS ====================
print("[Bonus 7/6] Creating Distribution Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#ffffff')

metrics_to_plot = ['redundancy_rate', 'omission_rate', 'hallucination_rate']
titles = ['Redundancy Rate Distribution', 'Omission Rate Distribution', 'Hallucination Rate Distribution']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx]
    baseline_data = results_df[results_df['model'] == 'Baseline'][metric].values
    aims_data = results_df[results_df['model'] == 'AIMS (Proposed)'][metric].values
    
    parts = ax.violinplot([baseline_data, aims_data], positions=[0, 1], widths=0.7, 
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#B0E0E6')
        pc.set_alpha(0.7)
    
    ax.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'AIMS'], fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    baseline_mean = np.mean(baseline_data)
    aims_mean = np.mean(aims_data)
    improvement = (baseline_mean - aims_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
    
    stats_text = f'Baseline: μ={baseline_mean:.3f}\nAIMS: μ={aims_mean:.3f}\nImprovement: {improvement:+.1f}%'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#FFF9E6', alpha=0.9, edgecolor='#FF6F00', linewidth=1.5))

fig.suptitle('Distribution Analysis: Baseline vs AIMS Performance', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('data/processed/07_distribution_analysis.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 07_distribution_analysis.png")
plt.close()

# ==================== IMAGE 8: CATEGORY PERFORMANCE HEATMAP ====================
print("[Bonus 8/6] Creating Category Performance Heatmap...")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#ffffff')

heatmap_data = []
for _, row in category_df.iterrows():
    baseline_score = (row['Baseline_Redundancy'] + row['Baseline_Omission'] + row['Baseline_Hallucination']) / 3
    aims_score = (row['AIMS_Redundancy'] + row['AIMS_Omission'] + row['AIMS_Hallucination']) / 3
    improvement = (baseline_score - aims_score) / baseline_score * 100 if baseline_score > 0 else 0
    heatmap_data.append(improvement)

categories = category_df['Category'].values
colors_map = ['#90EE90' if x > 0 else '#FFB6C6' for x in heatmap_data]
bars = ax.barh(categories, heatmap_data, color=colors_map, alpha=0.85, edgecolor=['#008000' if x > 0 else '#CC0000' for x in heatmap_data], linewidth=2.5)

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('AIMS Overall Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('Category-wise Overall Performance: AIMS vs Baseline\n(Combines Redundancy, Omission, and Hallucination Rates)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1)

# Add value labels
for bar, val in zip(bars, heatmap_data):
    x_pos = val + (2 if val > 0 else -2)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2., f'{val:+.1f}%',
            ha=ha, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('data/processed/08_category_heatmap.png', dpi=400, bbox_inches='tight', facecolor='#ffffff')
print("[OK] Image saved: 08_category_heatmap.png")
plt.close()

# ==================== SUMMARY ====================
print("\n" + "=" * 100)
print("IMAGE GENERATION COMPLETE")
print("=" * 100)

summary = """
Generated Images:
================

1. 01_metrics_comparison.png
   - Overall comparison of all three error metrics
   - Shows Baseline vs AIMS side-by-side
   
2. 02_redundancy_rate.png
   - Dedicated analysis of redundancy/repeated content
   - Includes calculation methodology
   
3. 03_omission_rate.png
   - Dedicated analysis of missing named entities
   - Shows entity coverage performance
   - Highlights where AIMS performs better
   
4. 04_hallucination_rate.png
   - Dedicated analysis of out-of-context entities
   - Shows factuality/accuracy metrics
   
5. 05_improvement_percentage.png
   - Overall improvement comparison
   - Shows which metrics favor AIMS vs Baseline
   - Green = AIMS Better, Red = Baseline Better
   
6. 06_category_omission_comparison.png
   - Omission rate by news category
   - Shows performance across 12 categories
   
7. 07_distribution_analysis.png
   - Violin plots for all three metrics
   - Shows data distribution and statistics
   
8. 08_category_heatmap.png
   - Overall category-wise performance
   - Combined score improvement visualization

All images are:
  ✓ High resolution (400 DPI)
  ✓ Professional formatting
  ✓ Clear titles and labels
  ✓ Color-coded for easy interpretation
  ✓ Include statistical annotations
  ✓ Ready for presentations and publications
"""

print(summary)

print("\n[Location] All images saved in: data/processed/")
print("[Format] PNG, 400 DPI, optimized for printing and presentations")
print("=" * 100)
