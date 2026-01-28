import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression\nRatio']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# Setup figure with proper dimensions for publication
fig = plt.figure(figsize=(14, 8), dpi=300)
ax = fig.add_subplot(111)

# X-axis positions
x = np.arange(len(metrics))
width = 0.35
offset = width / 2

# Create bars with professional colors
bars1 = ax.bar(x - offset, baseline, width, 
               label='Baseline', 
               color='#7F8C8D',      # Professional gray
               edgecolor='black', 
               linewidth=1.3, 
               alpha=0.9)

bars2 = ax.bar(x + offset, aims, width, 
               label='AIMS (Proposed)', 
               color='#2471A3',       # Professional blue
               edgecolor='black', 
               linewidth=1.3, 
               alpha=0.9)

# ============================================================================
# CUSTOMIZE AXES
# ============================================================================

# Y-axis
ax.set_ylabel('Score', fontsize=13, fontweight='bold', labelpad=12)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)

# X-axis
ax.set_xlabel('Evaluation Metrics', fontsize=13, fontweight='bold', labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')

# ============================================================================
# ADD VALUE LABELS ON BARS
# ============================================================================

def add_bar_labels(bars, values):
    """Add value labels on top of bars with proper positioning"""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = height + 0.018
        
        ax.text(x_pos, y_pos, f'{value:.2f}',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='black')

add_bar_labels(bars1, baseline)
add_bar_labels(bars2, aims)

# ============================================================================
# ADD GRID
# ============================================================================

ax.yaxis.grid(True, linestyle='-', alpha=0.25, linewidth=0.9, color='gray')
ax.set_axisbelow(True)
ax.xaxis.grid(False)

# ============================================================================
# LEGEND
# ============================================================================

legend = ax.legend(loc='upper left', 
                   fontsize=12, 
                   frameon=True,
                   fancybox=False,
                   edgecolor='black',
                   framealpha=1.0,
                   labelspacing=0.8,
                   handlelength=2)
legend.get_frame().set_linewidth(1.2)

# ============================================================================
# TITLE
# ============================================================================

title_text = ('Aggregate quantitative comparison between baseline and AIMS\n'
              'across lexical, semantic, and factual evaluation metrics')
ax.set_title(title_text, 
             fontsize=13, 
             fontweight='bold', 
             pad=25,
             linespacing=1.6)

# ============================================================================
# SPINE STYLING
# ============================================================================

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(1.3)
    ax.spines[spine].set_color('black')

# ============================================================================
# TICK STYLING
# ============================================================================

ax.tick_params(axis='y', which='major', length=6, width=1.2, labelsize=11)
ax.tick_params(axis='x', which='major', length=6, width=1.2, labelsize=11)

# ============================================================================
# ADJUST LAYOUT AND SAVE
# ============================================================================

# Adjust subplot parameters for better spacing
plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)

# Save as PNG (raster format)
plt.savefig('figure15.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.3)

print('✓ Figure saved as: figure15.png')
print(f'  Resolution: 300 DPI')
print(f'  Size: {os.path.getsize("figure15.png")/1024:.1f} KB')

# Save as PDF (vector format for journals)
plt.savefig('figure15.pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.3)

print('✓ Figure saved as: figure15.pdf')
print(f'  Format: Vector (scalable)')

# Save as EPS (PostScript for journals)
plt.savefig('figure15.eps', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.3)

print('✓ Figure saved as: figure15.eps')

# Display
plt.show()

print('\n' + '='*70)
print('FIGURE DETAILS')
print('='*70)
print('\nMetric Comparison:')
print('-' * 70)
print(f'{"Metric":<20} {"Baseline":>15} {"AIMS":>15} {"Improvement":>15}')
print('-' * 70)

metrics_short = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression']
for metric, base, aim in zip(metrics_short, baseline, aims):
    improvement = ((aim - base) / base * 100)
    if metric == 'Compression':
        improvement_str = f'{improvement:.1f}% (lower)'
    else:
        improvement_str = f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%'
    print(f'{metric:<20} {base:>15.4f} {aim:>15.4f} {improvement_str:>15}')

print('-' * 70)
print('\n✅ Publication-ready figures generated successfully!')
print('   Ready for journal submission (IEEE, ACL, EMNLP, etc.)')
