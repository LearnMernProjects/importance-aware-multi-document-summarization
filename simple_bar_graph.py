import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# Create figure - VERY LARGE
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)

# X positions
x = np.arange(len(metrics))
width = 0.4

# Create BARS
rects1 = ax.bar(x - width/2, baseline, width, label='Baseline', 
                color='#FF4444', edgecolor='black', linewidth=3)
rects2 = ax.bar(x + width/2, aims, width, label='AIMS (Proposed)', 
                color='#44DD44', edgecolor='black', linewidth=3)

# LABELS ON BARS - HUGE AND CLEAR
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.06,
            f'{height:.2f}', ha='center', va='bottom', fontsize=18, 
            fontweight='bold', color='black')

for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.06,
            f'{height:.2f}', ha='center', va='bottom', fontsize=18, 
            fontweight='bold', color='black')

# Y-AXIS
ax.set_ylabel('Score', fontsize=20, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18, fontweight='bold')

# X-AXIS
ax.set_xlabel('Metrics', fontsize=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=17, fontweight='bold')

# TITLE - HUGE
ax.set_title('Baseline vs AIMS - Performance Comparison', 
             fontsize=24, fontweight='bold', pad=30)

# LEGEND - HUGE
ax.legend(fontsize=18, loc='upper left', frameon=True, 
          edgecolor='black', fancybox=False, shadow=False)

# GRID
ax.grid(axis='y', alpha=0.3, linewidth=2, linestyle='--')
ax.set_axisbelow(True)

# TICK SIZE
ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)

# SPINES
for spine in ax.spines.values():
    spine.set_linewidth(3)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('figure15.png', dpi=300, bbox_inches='tight', facecolor='white')
print('✅ FIGURE SAVED: figure15.png')
print('   Location: c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\figure15.png')
print('\n✅ Features:')
print('   ✓ Figure size: 20x12 inches')
print('   ✓ Font size: 18-24pt (HUGE)')
print('   ✓ Bright RED and GREEN bars')
print('   ✓ Bold black borders')
print('   ✓ Clear numbers on bars')
print('   ✓ Large legend')
print('   ✓ 300 DPI quality')

plt.show()
