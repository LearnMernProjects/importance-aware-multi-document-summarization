"""
Proposed Importance-Aware Multi-Document Summarization Pipeline Diagram
Creates a publication-ready methodology visualization for the research paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CREATE FIGURE
# ============================================================
print("\n" + "="*70)
print("CREATING METHODOLOGY PIPELINE DIAGRAM")
print("="*70)

fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

# Define colors (light pastel, journal-friendly)
color_input = '#E8F4F8'      # Light blue
color_clustering = '#F0E8F4'  # Light purple
color_encoding = '#F4F0E8'    # Light beige
color_scoring = '#E8F4E8'     # Light green (novel component)
color_ordering = '#F4E8E8'    # Light pink
color_output = '#E8E8F4'      # Light lavender

# Box dimensions
box_width = 1.3
box_height = 1.2
y_center = 1.5

# ============================================================
# BLOCK 1: INPUT ARTICLES
# ============================================================
x1 = 0.3
box1 = FancyBboxPatch(
    (x1, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#2C3E50',
    facecolor=color_input,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box1)

ax.text(x1 + box_width/2, y_center + 0.15, 'Input News Articles',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x1 + box_width/2, y_center - 0.25, '{d₁, d₂, …, dₙ}',
        ha='center', va='center', fontsize=9, style='italic')

# ============================================================
# BLOCK 2: EVENT-LEVEL CLUSTERING
# ============================================================
x2 = 2.0
box2 = FancyBboxPatch(
    (x2, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#2C3E50',
    facecolor=color_clustering,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box2)

ax.text(x2 + box_width/2, y_center + 0.25, 'Event-level',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x2 + box_width/2, y_center - 0.05, 'Clustering',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x2 + box_width/2, y_center - 0.35, '(Temporal + Category\n+ Semantic Sim.)',
        ha='center', va='center', fontsize=8, style='italic')

# ============================================================
# BLOCK 3: ARTICLE ENCODING 
# ============================================================
x3 = 3.7
box3 = FancyBboxPatch(
    (x3, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#2C3E50',
    facecolor=color_encoding,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box3)

ax.text(x3 + box_width/2, y_center + 0.15, 'Article Encoding',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x3 + box_width/2, y_center - 0.25, 'hᵢ = Encoder(dᵢ)',
        ha='center', va='center', fontsize=9, style='italic', color='#1A5276')

# ============================================================
# BLOCK 4: IMPORTANCE SCORING (NOVEL)
# ============================================================
x4 = 5.4
box4 = FancyBboxPatch(
    (x4, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#27AE60',
    facecolor=color_scoring,
    linewidth=3.5,  # Thicker for emphasis
    zorder=2
)
ax.add_patch(box4)

# Add "NOVEL" badge
badge = mpatches.FancyBboxPatch(
    (x4 + box_width - 0.3, y_center + box_height/2 - 0.15),
    0.35,
    0.2,
    boxstyle="round,pad=0.02",
    edgecolor='#27AE60',
    facecolor='#27AE60',
    linewidth=1.5,
    zorder=3
)
ax.add_patch(badge)
ax.text(x4 + box_width - 0.125, y_center + box_height/2 - 0.05, 'NOVEL',
        ha='center', va='center', fontsize=7, fontweight='bold', color='white')

ax.text(x4 + box_width/2, y_center + 0.15, 'Importance Scoring',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#27AE60')
ax.text(x4 + box_width/2, y_center - 0.15, 'αᵢ = f(hᵢ)',
        ha='center', va='center', fontsize=9, style='italic', color='#27AE60')
ax.text(x4 + box_width/2, y_center - 0.35, 'wᵢ = softmax(αᵢ)',
        ha='center', va='center', fontsize=9, style='italic', color='#27AE60')

# ============================================================
# BLOCK 5: IMPORTANCE-AWARE ORDERING
# ============================================================
x5 = 7.1
box5 = FancyBboxPatch(
    (x5, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#2C3E50',
    facecolor=color_ordering,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box5)

ax.text(x5 + box_width/2, y_center + 0.15, 'Importance-aware',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x5 + box_width/2, y_center - 0.05, 'Ordering',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x5 + box_width/2, y_center - 0.35, 'Sort by wᵢ (descending)',
        ha='center', va='center', fontsize=8, style='italic')

# ============================================================
# BLOCK 6: SUMMARIZATION OUTPUT
# ============================================================
x6 = 8.8
box6 = FancyBboxPatch(
    (x6, y_center - box_height/2),
    box_width,
    box_height,
    boxstyle="round,pad=0.1",
    edgecolor='#2C3E50',
    facecolor=color_output,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(box6)

ax.text(x6 + box_width/2, y_center + 0.15, 'Summarization',
        ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(x6 + box_width/2, y_center - 0.15, 'BART / LLM',
        ha='center', va='center', fontsize=9, style='italic')
ax.text(x6 + box_width/2, y_center - 0.35, '→ Summary S',
        ha='center', va='center', fontsize=8, style='italic')

# ============================================================
# DRAW ARROWS CONNECTING BLOCKS
# ============================================================
arrow_y = y_center
arrow_height = 0.15

# Arrow 1-2
arrow1 = FancyArrowPatch(
    (x1 + box_width, arrow_y),
    (x2, arrow_y),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow1)

# Arrow 2-3
arrow2 = FancyArrowPatch(
    (x2 + box_width, arrow_y),
    (x3, arrow_y),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow2)

# Arrow 3-4
arrow3 = FancyArrowPatch(
    (x3 + box_width, arrow_y),
    (x4, arrow_y),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#27AE60',
    linewidth=3,
    zorder=1
)
ax.add_patch(arrow3)

# Arrow 4-5
arrow4 = FancyArrowPatch(
    (x4 + box_width, arrow_y),
    (x5, arrow_y),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#27AE60',
    linewidth=3,
    zorder=1
)
ax.add_patch(arrow4)

# Arrow 5-6
arrow5 = FancyArrowPatch(
    (x5 + box_width, arrow_y),
    (x6, arrow_y),
    arrowstyle='->,head_width=0.3,head_length=0.3',
    color='#34495E',
    linewidth=2.5,
    zorder=1
)
ax.add_patch(arrow5)

# ============================================================
# ADD TITLE
# ============================================================
ax.text(5, 2.85, 'Proposed Importance-Aware Multi-Document Summarization Pipeline',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#1A1A1A')

# ============================================================
# ADD LEGEND / DESCRIPTION
# ============================================================
legend_x = 0.3
legend_y = 0.25

ax.text(legend_x, legend_y + 0.3, 'Key Components:',
        fontsize=10, fontweight='bold', color='#2C3E50')

ax.text(legend_x, legend_y + 0.05, '• hᵢ: Semantic representation of article i',
        fontsize=8, color='#34495E')

ax.text(legend_x + 2.5, legend_y + 0.05, '• αᵢ: Unnormalized importance (centrality) score',
        fontsize=8, color='#34495E')

ax.text(legend_x, legend_y - 0.15, '• wᵢ: Normalized importance weight via softmax',
        fontsize=8, color='#27AE60', fontweight='bold')

ax.text(legend_x + 2.5, legend_y - 0.15, '• Σᵢ wᵢ = 1 (constraint satisfied)',
        fontsize=8, color='#27AE60')

# ============================================================
# SAVE FIGURE
# ============================================================
print("Saving figure...")

output_file = 'data/processed/proposed_method_pipeline.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"[OK] Figure saved: {output_file}")
print(f"  DPI: 300")
print(f"  Format: PNG")
print(f"  Size: Publication-ready for journal submissions")

# ============================================================
# DISPLAY CONFIRMATION
# ============================================================
print("\n" + "="*70)
print("FIGURE CREATION COMPLETE")
print("="*70)

print(f"\nPipeline Stages:")
print(f"  1. Input News Articles - Multi-document collection")
print(f"  2. Event-level Clustering - Temporal, category, and semantic grouping")
print(f"  3. Article Encoding - Sentence-transformers embeddings (hᵢ)")
print(f"  4. Importance Scoring - Novel centrality-based scoring (αᵢ, wᵢ) [NOVEL]")
print(f"  5. Importance-aware Ordering - Sort articles by importance weight")
print(f"  6. Summarization - BART/LLM generates final summary")

print(f"\nOutput File:")
print(f"  {output_file}")
print(f"  Ready for research paper (Methodology section)")

print(f"\nMathematical Notation Included:")
print(f"  ✓ hᵢ = Encoder(dᵢ)")
print(f"  ✓ αᵢ = f(hᵢ)")
print(f"  ✓ wᵢ = softmax(αᵢ)")
