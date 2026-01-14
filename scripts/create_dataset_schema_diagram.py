"""
NewsSumm Dataset Preparation and Schema Diagram
Creates a publication-ready visualization of the dataset pipeline
for the research paper dataset description section
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CREATE FIGURE
# ============================================================
print("\n" + "="*70)
print("CREATING NEWSSUMM DATASET SCHEMA DIAGRAM")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors (light pastel, journal-friendly)
color_raw = '#FFE8E8'        # Light red
color_cleaning = '#F0E8FF'   # Light purple
color_cleaned = '#E8F4FF'    # Light blue
color_ready = '#E8FFE8'      # Light green

# ============================================================
# BOX 1: RAW NEWSSUMM DATASET
# ============================================================
print("Drawing Box 1: Raw Dataset...")

box_width = 3.5
box_height = 1.2
x1 = 1.5
y1 = 8.0

box1 = FancyBboxPatch(
    (x1, y1),
    box_width,
    box_height,
    boxstyle="round,pad=0.15",
    edgecolor='#34495E',
    facecolor=color_raw,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box1)

ax.text(x1 + box_width/2, y1 + 0.8, 'Raw NewsSumm Dataset',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1A1A1A')
ax.text(x1 + box_width/2, y1 + 0.35, '(Excel / Raw Format)',
        ha='center', va='center', fontsize=10, style='italic', color='#34495E')

# ============================================================
# ARROW 1 -> 2
# ============================================================
arrow1 = FancyArrowPatch(
    (x1 + box_width/2, y1),
    (x1 + box_width/2, y1 - 1.0),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow1)

# ============================================================
# BOX 2: DATA CLEANING & VALIDATION
# ============================================================
print("Drawing Box 2: Cleaning & Validation...")

x2 = x1
y2 = y1 - 2.2
box_height2 = 1.5

box2 = FancyBboxPatch(
    (x2, y2),
    box_width,
    box_height2,
    boxstyle="round,pad=0.15",
    edgecolor='#8E44AD',
    facecolor=color_cleaning,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box2)

ax.text(x2 + box_width/2, y2 + 1.15, 'Data Cleaning & Validation',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#8E44AD')

cleaning_items = [
    '• Remove empty columns',
    '• Remove invalid rows',
    '• Normalize text'
]

for i, item in enumerate(cleaning_items):
    ax.text(x2 + box_width/2, y2 + 0.7 - i*0.3, item,
            ha='center', va='center', fontsize=9, color='#34495E')

# ============================================================
# ARROW 2 -> 3
# ============================================================
arrow2 = FancyArrowPatch(
    (x2 + box_width/2, y2),
    (x2 + box_width/2, y2 - 1.0),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow2)

# ============================================================
# BOX 3: CLEANED DATASET
# ============================================================
print("Drawing Box 3: Cleaned Dataset...")

x3 = x1
y3 = y2 - 2.4
box_height3 = 2.0

box3 = FancyBboxPatch(
    (x3, y3),
    box_width,
    box_height3,
    boxstyle="round,pad=0.15",
    edgecolor='#2980B9',
    facecolor=color_cleaned,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box3)

ax.text(x3 + box_width/2, y3 + 1.75, 'Cleaned Dataset',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#2980B9')

dataset_fields = [
    '• newspaper_name',
    '• published_date',
    '• headline',
    '• article_text',
    '• human_summary',
    '• news_category'
]

for i, field in enumerate(dataset_fields):
    ax.text(x3 + 0.5, y3 + 1.35 - i*0.25, field,
            ha='left', va='center', fontsize=8.5, color='#1A1A1A', family='monospace')

# ============================================================
# ARROW 3 -> 4
# ============================================================
arrow3 = FancyArrowPatch(
    (x3 + box_width/2, y3),
    (x3 + box_width/2, y3 - 0.9),
    arrowstyle='->,head_width=0.4,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow3)

# ============================================================
# BOX 4: CLUSTERING-READY DATASET
# ============================================================
print("Drawing Box 4: Clustering-Ready Dataset...")

x4 = x1
y4 = y3 - 1.6
box_height4 = 1.2

box4 = FancyBboxPatch(
    (x4, y4),
    box_width,
    box_height4,
    boxstyle="round,pad=0.15",
    edgecolor='#27AE60',
    facecolor=color_ready,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box4)

ax.text(x4 + box_width/2, y4 + 0.8, 'Prepared for',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#27AE60')
ax.text(x4 + box_width/2, y4 + 0.45, 'Event-level Clustering',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#27AE60')
ax.text(x4 + box_width/2, y4 + 0.05, '(Multi-document Construction)',
        ha='center', va='center', fontsize=9, style='italic', color='#1A1A1A')

# ============================================================
# RIGHT SIDE: DATASET STATISTICS BOX
# ============================================================
print("Drawing Statistics Box...")

x_stats = 5.5
y_stats = 6.5
width_stats = 3.8
height_stats = 3.2

stats_box = FancyBboxPatch(
    (x_stats, y_stats),
    width_stats,
    height_stats,
    boxstyle="round,pad=0.15",
    edgecolor='#F39C12',
    facecolor='#FEF5E7',
    linewidth=2,
    zorder=2,
    linestyle='--'
)
ax.add_patch(stats_box)

ax.text(x_stats + width_stats/2, y_stats + height_stats - 0.35, 'Dataset Statistics',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#F39C12')

stats_data = [
    'Total Articles: 348,766',
    'After Cleaning: 346,877',
    'Removed: 1,889 rows',
    '',
    'Multi-doc Clusters: 27',
    'Articles in Clusters: 62',
    'Avg. Cluster Size: 2.3'
]

for i, stat in enumerate(stats_data):
    if stat:  # Skip empty lines
        ax.text(x_stats + 0.3, y_stats + height_stats - 0.85 - i*0.35, stat,
                ha='left', va='center', fontsize=9, color='#1A1A1A', family='monospace')
    else:
        # Draw a separator line
        ax.plot([x_stats + 0.2, x_stats + width_stats - 0.2], 
                [y_stats + height_stats - 0.85 - i*0.35, y_stats + height_stats - 0.85 - i*0.35],
                'k-', linewidth=0.5, alpha=0.3)

# ============================================================
# RIGHT SIDE: DATA QUALITY METRICS BOX
# ============================================================
print("Drawing Data Quality Box...")

x_quality = 5.5
y_quality = 2.0
width_quality = 3.8
height_quality = 3.8

quality_box = FancyBboxPatch(
    (x_quality, y_quality),
    width_quality,
    height_quality,
    boxstyle="round,pad=0.15",
    edgecolor='#E74C3C',
    facecolor='#FADBD8',
    linewidth=2,
    zorder=2,
    linestyle='--'
)
ax.add_patch(quality_box)

ax.text(x_quality + width_quality/2, y_quality + height_quality - 0.35, 'Data Quality Metrics',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#E74C3C')

quality_data = [
    'Missing Values Removed:',
    '  • Invalid dates: 1,351',
    '  • Missing articles: 23',
    '  • Short articles: 522',
    '',
    'Text Normalization:',
    '  • Lowercased',
    '  • Extra whitespace removed',
    '  • Encoding: UTF-8'
]

for i, item in enumerate(quality_data):
    if item and not item.startswith('  '):
        ax.text(x_quality + 0.3, y_quality + height_quality - 0.85 - i*0.35, item,
                ha='left', va='center', fontsize=9, fontweight='bold', color='#1A1A1A')
    elif item.startswith('  '):
        ax.text(x_quality + 0.5, y_quality + height_quality - 0.85 - i*0.35, item,
                ha='left', va='center', fontsize=8.5, color='#34495E', family='monospace')
    else:
        # Separator line
        ax.plot([x_quality + 0.2, x_quality + width_quality - 0.2], 
                [y_quality + height_quality - 0.85 - i*0.35, y_quality + height_quality - 0.85 - i*0.35],
                'k-', linewidth=0.5, alpha=0.3)

# ============================================================
# ADD TITLE
# ============================================================
print("Adding title...")

ax.text(5, 9.5, 'NewsSumm Dataset Preparation and Schema',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#1A1A1A')

# ============================================================
# ADD FOOTER
# ============================================================
ax.text(5, 0.5, 'Dataset: 346,877 articles from 4 Indian English newspapers (1950-2020)',
        ha='center', va='center', fontsize=9, style='italic', color='#7F8C8D')

# ============================================================
# SAVE FIGURE
# ============================================================
print("\nSaving figure...")

output_file = 'data/processed/newssumm_dataset_schema.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"[OK] Figure saved: {output_file}")
print(f"  DPI: 300")
print(f"  Format: PNG")
print(f"  Size: Publication-ready for journal submissions")

plt.close()

# ============================================================
# DISPLAY CONFIRMATION
# ============================================================
print("\n" + "="*70)
print("FIGURE CREATION COMPLETE")
print("="*70)

print(f"\nDataset Pipeline Shown:")
print(f"  1. Raw NewsSumm Dataset (Excel/Raw Format)")
print(f"  2. Data Cleaning & Validation")
print(f"     - Remove empty columns")
print(f"     - Remove invalid rows")
print(f"     - Normalize text")
print(f"  3. Cleaned Dataset with 6 core fields")
print(f"  4. Prepared for Event-level Clustering")

print(f"\nIncluded Information:")
print(f"  ✓ Dataset schema (field names)")
print(f"  ✓ Data quality metrics")
print(f"  ✓ Cleaning statistics")
print(f"  ✓ Multi-document clustering readiness")

print(f"\nOutput File:")
print(f"  {output_file}")
print(f"  Ready for research paper (Dataset Description section)")
