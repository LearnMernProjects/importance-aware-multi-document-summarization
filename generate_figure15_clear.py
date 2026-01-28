import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# ============================================================================
# CLEAR AND LARGE FIGURE
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10), dpi=300)

# X-axis positions
x = np.arange(len(metrics))
width = 0.38
offset = width / 2

# Create bars with HIGHLY VISIBLE colors
bars1 = ax.bar(x - offset, baseline, width, 
               label='Baseline', 
               color='#E74C3C',      # Bright red
               edgecolor='#000000', 
               linewidth=2.5, 
               alpha=0.95)

bars2 = ax.bar(x + offset, aims, width, 
               label='AIMS (Proposed)', 
               color='#27AE60',       # Bright green
               edgecolor='#000000', 
               linewidth=2.5, 
               alpha=0.95)

# ============================================================================
# LARGE AND CLEAR LABELS
# ============================================================================

ax.set_ylabel('Score', fontsize=16, fontweight='bold', labelpad=15)
ax.set_ylim(0, 1.20)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14, fontweight='bold')

ax.set_xlabel('Evaluation Metrics', fontsize=16, fontweight='bold', labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13, fontweight='bold')

# ============================================================================
# ADD VALUE LABELS - LARGE AND CLEAR
# ============================================================================

def add_bar_labels(bars, values):
    """Add large, clear value labels"""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = height + 0.08
        
        # Large, bold, black text with white background for clarity
        ax.text(x_pos, y_pos, f'{value:.2f}',
                ha='center', 
                va='bottom',
                fontsize=13,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='white', 
                         edgecolor='black',
                         linewidth=1.5,
                         alpha=0.95))

add_bar_labels(bars1, baseline)
add_bar_labels(bars2, aims)

# ============================================================================
# GRID
# ============================================================================

ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=1.5, color='gray')
ax.set_axisbelow(True)
ax.xaxis.grid(False)

# ============================================================================
# LEGEND - LARGE AND CLEAR
# ============================================================================

legend = ax.legend(loc='upper left', 
                   fontsize=14, 
                   frameon=True,
                   fancybox=False,
                   edgecolor='black',
                   framealpha=0.95,
                   labelspacing=1.2,
                   handlelength=2.5)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_facecolor('white')

# ============================================================================
# TITLE - LARGE AND CLEAR
# ============================================================================

title_text = ('Aggregate Quantitative Comparison Between Baseline and AIMS\n'
              'Across Lexical, Semantic, and Factual Evaluation Metrics')
ax.set_title(title_text, 
             fontsize=15, 
             fontweight='bold', 
             pad=30,
             linespacing=1.8)

# ============================================================================
# SPINE STYLING
# ============================================================================

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(2.5)
    ax.spines[spine].set_color('black')

# ============================================================================
# TICK STYLING - LARGER
# ============================================================================

ax.tick_params(axis='y', which='major', length=8, width=2, labelsize=13, colors='black')
ax.tick_params(axis='x', which='major', length=8, width=2, labelsize=13, colors='black')

# ============================================================================
# WHITE BACKGROUND
# ============================================================================

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ============================================================================
# ADJUST AND SAVE
# ============================================================================

plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.10)

# Save PNG
plt.savefig('figure15.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.5)

print('✓ CLEAR figure saved as: figure15.png')
print(f'  Size: {os.path.getsize("figure15.png")/1024:.1f} KB')

# Save PDF
plt.savefig('figure15.pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.5)

print('✓ CLEAR figure saved as: figure15.pdf')

plt.show()

print('\n' + '='*70)
print('FIGURE CREATED - NOW CRYSTAL CLEAR!')
print('='*70)
print('\n✅ Enhanced Clarity Features:')
print('   • Larger figure size (16x10 inches)')
print('   • Bright colors (Red for Baseline, Green for AIMS)')
print('   • Large fonts (13-16pt)')
print('   • Bold text throughout')
print('   • White background boxes around numbers')
print('   • Black borders on all elements')
print('   • Clear spacing between elements')
print('   • High contrast for easy reading')
print('\n' + '='*70)
