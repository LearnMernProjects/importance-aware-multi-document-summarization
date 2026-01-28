import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore-F1', 'Faithfulness', 'Compression']
baseline = [0.47, 0.30, 0.44, 0.89, 0.93, 0.38]
aims = [0.52, 0.35, 0.50, 0.92, 0.95, 0.36]

# Create figure - VERY LARGE with proper margins
fig = plt.figure(figsize=(20, 14))
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
ax.set_ylabel('Score', fontsize=22, fontweight='bold', labelpad=20)
ax.set_ylim(0, 1.1)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18, fontweight='bold')

# X-AXIS - WITH EXTRA SPACE
ax.set_xlabel('Metrics', fontsize=22, fontweight='bold', labelpad=30)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18, fontweight='bold')

# TITLE - HUGE WITH PADDING
ax.set_title('Baseline vs AIMS - Performance Comparison', 
             fontsize=26, fontweight='bold', pad=50)

# LEGEND - HUGE
ax.legend(fontsize=20, loc='upper left', frameon=True, 
          edgecolor='black', fancybox=False, shadow=False)

# GRID
ax.grid(axis='y', alpha=0.3, linewidth=2, linestyle='--')
ax.set_axisbelow(True)

# TICK SIZE
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=10)

# SPINES - THICK AND VISIBLE
for spine in ax.spines.values():
    spine.set_linewidth(3)
    spine.set_color('black')

# PROPER MARGINS - VERY IMPORTANT!
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

# SAVE WITH PADDING
plt.savefig('figure15.png', dpi=300, bbox_inches='tight', 
            facecolor='white', pad_inches=0.8)
print('✅ FIGURE SAVED: figure15.png')
print('   Location: c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\figure15.png')
print('\n✅ Fixed Features:')
print('   ✓ Title FULLY VISIBLE (padding: 50)')
print('   ✓ X-axis label FULLY VISIBLE (padding: 30)')
print('   ✓ Y-axis label FULLY VISIBLE (padding: 20)')
print('   ✓ Figure size: 20x14 inches (EXTRA HEIGHT)')
print('   ✓ Margins: left=15%, right=95%, top=92%, bottom=15%')
print('   ✓ Font size: 18-26pt (HUGE)')
print('   ✓ Bright RED and GREEN bars')
print('   ✓ Bold black borders (3pt)')
print('   ✓ Clear numbers on bars')
print('   ✓ 300 DPI quality')
print('   ✓ Extra padding around figure (0.8 inches)')

plt.show()
