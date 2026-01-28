import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression Ratio']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# Setup
x = np.arange(len(metrics))
width = 0.35

# Create figure with high DPI
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Create bars
bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', 
               color='#999999', edgecolor='black', linewidth=1.2, alpha=0.85)
bars2 = ax.bar(x + width/2, aims, width, label='AIMS (Proposed)', 
               color='#2E86AB', edgecolor='black', linewidth=1.2, alpha=0.85)

# Customize axes
ax.set_ylabel('Score', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.05)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax.set_axisbelow(True)

# Add values on top of bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Legend
ax.legend(loc='upper right', fontsize=11, frameon=True, 
          fancybox=False, edgecolor='black', framealpha=1.0)

# Title
title_text = ('Aggregate quantitative comparison between baseline and AIMS\n'
              'across lexical, semantic, and factual evaluation metrics.')
ax.set_title(title_text, fontsize=13, fontweight='bold', pad=20, linespacing=1.5)

# Spine styling
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Y-axis limits and ticks
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'], fontsize=10)

# Tight layout
plt.tight_layout()

# Save with high quality
plt.savefig('figure15.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print('✓ Figure saved as: figure15.png (300 DPI, publication-quality)')
print(f'  File size: ~{os.path.getsize("figure15.png")/1024:.1f} KB')

# Also save as PDF for journals
plt.savefig('figure15.pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print('✓ PDF version saved as: figure15.pdf')

plt.show()
