import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# High-quality rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.antialiased'] = True

def draw_3d_box(ax, x, y, z, size, color, alpha=0.8, label=''):
    """Draw a 3D box/cube with labels"""
    r = [-size/2, size/2]
    X, Y = np.meshgrid(r, r)
    
    # Define the 6 faces of the cube
    vertices = [
        [[x+r[0], y+r[0], z+r[0]], [x+r[1], y+r[0], z+r[0]], [x+r[1], y+r[1], z+r[0]], [x+r[0], y+r[1], z+r[0]]],
        [[x+r[0], y+r[0], z+r[1]], [x+r[1], y+r[0], z+r[1]], [x+r[1], y+r[1], z+r[1]], [x+r[0], y+r[1], z+r[1]]],
        [[x+r[0], y+r[0], z+r[0]], [x+r[0], y+r[1], z+r[0]], [x+r[0], y+r[1], z+r[1]], [x+r[0], y+r[0], z+r[1]]],
        [[x+r[1], y+r[0], z+r[0]], [x+r[1], y+r[1], z+r[0]], [x+r[1], y+r[1], z+r[1]], [x+r[1], y+r[0], z+r[1]]],
        [[x+r[0], y+r[0], z+r[0]], [x+r[1], y+r[0], z+r[0]], [x+r[1], y+r[0], z+r[1]], [x+r[0], y+r[0], z+r[1]]],
        [[x+r[0], y+r[1], z+r[0]], [x+r[1], y+r[1], z+r[0]], [x+r[1], y+r[1], z+r[1]], [x+r[0], y+r[1], z+r[1]]]
    ]
    
    cube = Poly3DCollection(vertices, alpha=alpha, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_collection3d(cube)
    
    # Add text label
    ax.text(x, y, z+size/2+0.8, label, ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

def draw_arrow_3d(ax, x_start, y_start, z_start, x_end, y_end, z_end, color='black', width=0.15):
    """Draw 3D arrow between two points"""
    # Main line
    ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], 
           color=color, linewidth=3, zorder=5)

def create_clean_3d_pipeline():
    """
    Create a clean, understandable 3D pipeline visualization
    showing the multi-document summarization process step-by-step
    """
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#ffffff')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f0f0')
    
    print("Creating clean 3D pipeline visualization...")
    
    # ===== STAGE 1: MULTIPLE INPUT DOCUMENTS =====
    print("Stage 1: Input Documents")
    stage1_x = 0
    stage1_z = 0
    
    # Draw 5 document boxes in a row
    for i in range(5):
        y_pos = -2.2 + i * 1.1
        draw_3d_box(ax, stage1_x, y_pos, stage1_z, 0.7, '#4472C4', alpha=0.85, label=f'Doc {i+1}')
    
    # Stage 1 label
    ax.text(stage1_x-1.5, 0, stage1_z, 'STAGE 1:\nINPUT\nDOCUMENTS', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#B4C7E7', alpha=0.9, edgecolor='#1F4E78', linewidth=2))
    
    # ===== ARROW 1 =====
    print("Drawing arrow 1")
    draw_arrow_3d(ax, stage1_x+0.5, 0, stage1_z, 3.5, 0, 1.5, color='#333333', width=0.2)
    ax.text(2, 0.5, 1.8, '①', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ===== STAGE 2: TEXT PREPROCESSING =====
    print("Stage 2: Preprocessing")
    stage2_x = 3.5
    stage2_z = 1.5
    
    draw_3d_box(ax, stage2_x, 0, stage2_z, 1.2, '#70AD47', alpha=0.85, label='Preprocessing\n(Tokenization,\nCleaning)')
    
    ax.text(stage2_x-2, -1.5, stage2_z, 'STAGE 2:\nPREPROCESSING', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#C6E0B4', alpha=0.9, edgecolor='#375623', linewidth=2))
    
    # ===== ARROW 2 =====
    print("Drawing arrow 2")
    draw_arrow_3d(ax, stage2_x+0.65, 0, stage2_z+0.75, 6.5, 0, 3, color='#333333', width=0.2)
    ax.text(5.5, 0.5, 2.5, '②', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ===== STAGE 3: EVENT CLUSTERING =====
    print("Stage 3: Event Clustering")
    stage3_x = 6.5
    stage3_z = 3
    
    # Draw 4 cluster boxes
    cluster_positions = [-1.2, -0.4, 0.4, 1.2]
    for i, y_pos in enumerate(cluster_positions):
        draw_3d_box(ax, stage3_x, y_pos, stage3_z, 0.7, '#FFC000', alpha=0.85, label=f'Cluster\n{i+1}')
    
    ax.text(stage3_x-2.5, 0, stage3_z, 'STAGE 3:\nEVENT\nCLUSTERING', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEB9C', alpha=0.9, edgecolor='#997300', linewidth=2))
    
    # ===== ARROW 3 =====
    print("Drawing arrow 3")
    draw_arrow_3d(ax, stage3_x+0.5, 0, stage3_z+0.5, 9.5, 0, 4.2, color='#333333', width=0.2)
    ax.text(8.5, 0.6, 3.8, '③', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ===== STAGE 4: IMPORTANCE SCORING =====
    print("Stage 4: Importance Scoring")
    stage4_x = 9.5
    stage4_z = 4.2
    
    draw_3d_box(ax, stage4_x, 0, stage4_z, 1.3, '#ED7D31', alpha=0.85, label='Importance\nScoring\n(TF-IDF,\nRelevance)')
    
    ax.text(stage4_x-2.5, -1.5, stage4_z, 'STAGE 4:\nIMPORTANCE\nSCORING', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F4B084', alpha=0.9, edgecolor='#C65911', linewidth=2))
    
    # ===== ARROW 4 =====
    print("Drawing arrow 4")
    draw_arrow_3d(ax, stage4_x+0.7, 0, stage4_z+0.75, 12.5, 0, 5.5, color='#333333', width=0.2)
    ax.text(11.5, 0.6, 5, '④', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ===== STAGE 5: CONTENT SELECTION & AGGREGATION =====
    print("Stage 5: Content Selection")
    stage5_x = 12.5
    stage5_z = 5.5
    
    draw_3d_box(ax, stage5_x, 0, stage5_z, 1.4, '#A64D79', alpha=0.85, label='Content\nSelection &\nAggregation')
    
    ax.text(stage5_x-2.8, -1.5, stage5_z, 'STAGE 5:\nCONTENT\nSELECTION', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E2DFDE', alpha=0.9, edgecolor='#70567B', linewidth=2))
    
    # ===== ARROW 5 =====
    print("Drawing arrow 5")
    draw_arrow_3d(ax, stage5_x+0.75, 0, stage5_z+0.8, 15.5, 0, 6.5, color='#333333', width=0.2)
    ax.text(14.2, 0.7, 6, '⑤', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # ===== STAGE 6: FINAL SUMMARY OUTPUT =====
    print("Stage 6: Final Output")
    stage6_x = 15.5
    stage6_z = 6.5
    
    draw_3d_box(ax, stage6_x, 0, stage6_z, 1.5, '#70AD47', alpha=0.9, label='FINAL\nSUMMARY\nOUTPUT')
    
    # Add glow effect with larger transparent box
    draw_3d_box(ax, stage6_x, 0, stage6_z, 1.8, '#70AD47', alpha=0.2, label='')
    
    ax.text(stage6_x-2.8, -1.8, stage6_z, 'STAGE 6:\nOUTPUT\nSUMMARY', 
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#C6E0B4', alpha=0.9, edgecolor='#375623', linewidth=2))
    
    # ===== METRICS/QUALITY INDICATORS (Below the pipeline) =====
    print("Adding quality metrics")
    metrics_z = -1.5
    metrics_labels = ['ROUGE-L\n0.67', 'BERTScore\n0.88', 'METEOR\n0.71', 'Relevance\n0.85', 'Coherence\n0.82', 'Factuality\n0.79']
    x_positions = np.linspace(1.5, 14.5, len(metrics_labels))
    
    for i, (x_pos, metric) in enumerate(zip(x_positions, metrics_labels)):
        ax.scatter([x_pos], [0], [metrics_z], s=600, c='#9467bd', marker='D', 
                  edgecolors='#6B3A8C', linewidth=2, zorder=4, alpha=0.8)
        ax.text(x_pos, 0.8, metrics_z, metric, ha='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8, edgecolor='#6B3A8C'))
    
    # ===== STATISTICS BOX =====
    print("Adding statistics")
    stats_text = 'INPUT: 10 Documents\nCLUSTERS: 4 Events\nAVG SCORE: 0.81\nOUTPUT: 384 words'
    ax.text(1, -3.5, 7.5, stats_text, ha='left', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', alpha=0.95, 
                    edgecolor='#1F4E78', linewidth=2))
    
    # ===== FORMATTING =====
    print("Applying formatting")
    
    ax.set_xlabel('\nPipeline Progression →', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('\nDocuments/Events →', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_zlabel('\nProcessing Height →', fontsize=13, fontweight='bold', labelpad=15)
    
    ax.set_xlim(-2, 18)
    ax.set_ylim(-4.5, 3)
    ax.set_zlim(-2.5, 8)
    
    # Professional grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Optimal viewing angle
    ax.view_init(elev=25, azim=35)
    
    # ===== LEGEND =====
    print("Adding legend")
    legend_elements = [
        mpatches.Patch(facecolor='#4472C4', edgecolor='#1F4E78', linewidth=2, label='Stage 1: Input Documents (5 documents)'),
        mpatches.Patch(facecolor='#70AD47', edgecolor='#375623', linewidth=2, label='Stage 2: Preprocessing & Tokenization'),
        mpatches.Patch(facecolor='#FFC000', edgecolor='#997300', linewidth=2, label='Stage 3: Event Clustering (4 clusters)'),
        mpatches.Patch(facecolor='#ED7D31', edgecolor='#C65911', linewidth=2, label='Stage 4: Importance Scoring'),
        mpatches.Patch(facecolor='#A64D79', edgecolor='#70567B', linewidth=2, label='Stage 5: Content Selection'),
        mpatches.Patch(facecolor='#70AD47', edgecolor='#375623', linewidth=2, label='Stage 6: Final Summary Output'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
             framealpha=0.95, edgecolor='black', fancybox=True, shadow=True, ncol=1)
    legend.get_frame().set_linewidth(2)
    
    # ===== TITLE =====
    print("Adding title")
    title_text = 'Multi-Document Summarization Pipeline - 6 Stage Linear Process'
    subtitle_text = 'Importance-Aware Approach: Document → Cluster → Score → Select → Aggregate → Summary'
    
    fig.text(0.5, 0.98, title_text, ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8', alpha=0.95, edgecolor='#1F4E78', linewidth=2.5))
    fig.text(0.5, 0.945, subtitle_text, ha='center', fontsize=12, style='italic', color='#333333')
    
    # ===== SAVE =====
    print("Saving high-resolution image...")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_pipeline_clean.png', 
               dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none',
               pad_inches=0.3)
    
    print("✓ Clean 3D pipeline visualization saved as: 3d_pipeline_clean.png")
    print("✓ Resolution: 400 DPI (Professional Quality)")
    print("✓ Clear 6-stage linear pipeline")
    print("✓ Easy to understand flow for researchers")
    
    plt.show()

if __name__ == "__main__":
    create_clean_3d_pipeline()
