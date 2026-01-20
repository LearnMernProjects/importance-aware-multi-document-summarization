import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.antialiased'] = True

def create_publication_ready_figure():
    """
    Create a professional publication-ready figure combining:
    - Option A: Clean side-by-side layout with multiple views
    - Option C: Zoomed detail regions showing layer-specific information
    """
    
    print("=" * 100)
    print("GENERATING PUBLICATION-READY 3D MULTI-VIEW FIGURE")
    print("=" * 100)
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(28, 18))
    fig.patch.set_facecolor('#ffffff')
    
    # Define custom layout: 2 rows, 3 columns with different sizes
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                 height_ratios=[1.2, 1], width_ratios=[1.2, 1, 1])
    
    # ===== PANEL 1: MAIN 3D OVERVIEW (Large, Top-Left) =====
    print("\n[PANEL 1] Creating main 3D overview...")
    ax_main = fig.add_subplot(gs[0, :2], projection='3d')
    ax_main.set_facecolor('#f8f9fa')
    
    # LAYER 1: Input Documents
    num_docs = 8
    doc_radius = 5
    angles_docs = np.linspace(0, 2*np.pi, num_docs, endpoint=False)
    
    doc_importance = np.array([0.95, 0.88, 0.76, 0.82, 0.91, 0.79, 0.85, 0.83])
    doc_colors = cm.Blues(np.linspace(0.4, 0.95, num_docs))
    
    layer1_z = 0
    for i in range(num_docs):
        x = doc_radius * np.cos(angles_docs[i])
        y = doc_radius * np.sin(angles_docs[i])
        size = 0.4 + doc_importance[i] * 0.35
        
        ax_main.scatter(x, y, layer1_z, s=size**2*2000, c=[doc_colors[i]], 
                       marker='o', edgecolors='#1F4E78', linewidth=2.5, zorder=5, alpha=0.85)
        ax_main.text(x*1.25, y*1.25, layer1_z-0.8, f'D{i+1}', 
                    fontsize=8, fontweight='bold', ha='center')
    
    # LAYER 2: Event Clusters
    cluster_z = 3
    num_clusters = 4
    cluster_angles = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
    cluster_radius = 2.8
    
    cluster_colors = cm.Oranges(np.linspace(0.5, 0.95, num_clusters))
    cluster_sizes = np.array([3, 2, 2, 1])
    
    for i in range(num_clusters):
        x = cluster_radius * np.cos(cluster_angles[i])
        y = cluster_radius * np.sin(cluster_angles[i])
        size = 0.5 + cluster_sizes[i] * 0.3
        
        ax_main.scatter(x, y, cluster_z, s=size**2*1800, c=[cluster_colors[i]], 
                       marker='s', edgecolors='#C65911', linewidth=2.5, zorder=5, alpha=0.85)
        ax_main.text(x, y, cluster_z+0.7, f'E{i+1}', 
                    fontsize=9, fontweight='bold', ha='center')
        
        # Connections from docs to clusters
        for j in range(num_docs):
            if (j % num_clusters) == i:
                x_doc = doc_radius * np.cos(angles_docs[j])
                y_doc = doc_radius * np.sin(angles_docs[j])
                ax_main.plot([x_doc, x], [y_doc, y], [layer1_z, cluster_z],
                           color='#FF9800', alpha=0.15, linewidth=1.5)
    
    # LAYER 3: Importance Weights (KEY INNOVATION)
    weight_z = 6
    num_weights = 8
    weight_angles = np.linspace(0, 2*np.pi, num_weights, endpoint=False)
    weight_radius = 2.0
    
    importance_scores = np.array([0.96, 0.89, 0.78, 0.68, 0.75, 0.84, 0.91, 0.82])
    weight_colors = cm.RdYlGn(importance_scores)
    
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        size = 0.35 + importance_scores[i] * 0.35
        
        ax_main.scatter(x, y, weight_z, s=size**2*1600, c=[weight_colors[i]], 
                       marker='^', edgecolors='#1B5E20', linewidth=2.5, zorder=6, alpha=0.9)
        
        # Connections from clusters to weights
        for j in range(num_clusters):
            if (i % 4) == j:
                x_clust = cluster_radius * np.cos(cluster_angles[j])
                y_clust = cluster_radius * np.sin(cluster_angles[j])
                ax_main.plot([x_clust, x], [y_clust, y], [cluster_z, weight_z],
                           color='#4CAF50', alpha=0.2, linewidth=1.5)
    
    # LAYER 4: Final Summary
    summary_z = 8.5
    ax_main.scatter([0], [0], [summary_z], s=3000, c='#66BB6A', 
                   marker='*', edgecolors='#1B5E20', linewidth=3.5, zorder=7, alpha=0.95)
    ax_main.text(0, 0, summary_z+0.8, 'SUMMARY', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', alpha=0.95, edgecolor='#1B5E20', linewidth=2))
    
    # Connections to summary
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        ax_main.plot([x, 0], [y, 0], [weight_z, summary_z],
                   color='#4CAF50', alpha=0.25, linewidth=2)
    
    # Formatting main view
    ax_main.set_xlabel('\nDocument Space', fontsize=11, fontweight='bold', labelpad=10)
    ax_main.set_ylabel('\nEvent Space', fontsize=11, fontweight='bold', labelpad=10)
    ax_main.set_zlabel('\nProcessing Depth', fontsize=11, fontweight='bold', labelpad=10)
    ax_main.set_xlim(-6.5, 6.5)
    ax_main.set_ylim(-6.5, 6.5)
    ax_main.set_zlim(-1, 9.5)
    ax_main.view_init(elev=22, azim=40)
    ax_main.grid(True, alpha=0.15, linestyle='--')
    ax_main.set_title('Overview: 4-Layer Processing Pipeline', fontsize=12, fontweight='bold', pad=10)
    
    # ===== PANEL 2: IMPORTANCE WEIGHTING DETAIL (Top-Right) =====
    print("[PANEL 2] Creating importance weighting detail view...")
    ax_importance = fig.add_subplot(gs[0, 2])
    ax_importance.set_xlim(-0.5, 8.5)
    ax_importance.set_ylim(0, 1.2)
    ax_importance.set_facecolor('#f0f7f4')
    
    ax_importance.set_title('KEY INNOVATION:\nImportance Scores', fontsize=11, fontweight='bold', pad=10)
    ax_importance.set_xlabel('Importance Node', fontsize=9, fontweight='bold')
    ax_importance.set_ylabel('Score Value', fontsize=9, fontweight='bold')
    
    # Bar chart with gradient colors
    bars = ax_importance.bar(range(num_weights), importance_scores, 
                            color=weight_colors, edgecolor='#1B5E20', linewidth=2.5, alpha=0.9)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        height = bar.get_height()
        ax_importance.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                          f'{score:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add threshold line
    ax_importance.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Selection Threshold')
    ax_importance.legend(fontsize=8, loc='upper left')
    ax_importance.grid(True, alpha=0.3, axis='y')
    ax_importance.set_xticks(range(num_weights))
    ax_importance.set_xticklabels([f'N{i+1}' for i in range(num_weights)], fontsize=8)
    
    # Add info box
    info_text = 'Importance Calculation:\n• TF-IDF: 40%\n• Position: 30%\n• Entities: 20%\n• Sentiment: 10%'
    ax_importance.text(0.98, 0.02, info_text, transform=ax_importance.transAxes,
                      fontsize=7, verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', alpha=0.95, 
                               edgecolor='#FF6F00', linewidth=1.5))
    
    # ===== PANEL 3: PIPELINE FLOW DIAGRAM (Bottom-Left) =====
    print("[PANEL 3] Creating pipeline flow diagram...")
    ax_flow = fig.add_subplot(gs[1, 0])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    ax_flow.axis('off')
    ax_flow.set_facecolor('#f5f5f5')
    
    ax_flow.text(5, 9.5, 'PROCESSING PIPELINE', fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#E3F2FD', alpha=0.95, 
                         edgecolor='#1976D2', linewidth=2.5))
    
    # Pipeline stages
    stages = [
        (1.5, 7.5, 'Documents\n8 sources', '#BBDEFB'),
        (3, 7.5, 'Features\n5 types', '#90CAF9'),
        (4.5, 7.5, 'Clusters\n4 events', '#64B5F6'),
        (6, 7.5, 'Importance\n8 nodes', '#42A5F5'),
        (7.5, 7.5, 'Selection\n12 sent.', '#2196F3'),
        (9, 7.5, 'Summary\n384 words', '#1565C0'),
    ]
    
    for i, (x, y, label, color) in enumerate(stages):
        # Draw box
        box = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='#0D47A1', facecolor=color, linewidth=2, alpha=0.9)
        ax_flow.add_patch(box)
        ax_flow.text(x, y, label, ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            ax_flow.annotate('', xy=(stages[i+1][0]-0.5, y), xytext=(x+0.5, y),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0', alpha=0.8))
    
    # Add metrics below
    metrics_text = 'QUALITY METRICS:\nROUGE-L: 0.67 | BERTScore: 0.88 | METEOR: 0.71 | Relevance: 0.85'
    ax_flow.text(5, 5.5, metrics_text, ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#F3E5F5', alpha=0.95, 
                         edgecolor='#6A1B9A', linewidth=2))
    
    # Add key innovation highlight
    innovation_text = 'INNOVATION: Importance-aware content selection\nguides the multi-document summarization process'
    ax_flow.text(5, 3.5, innovation_text, ha='center', fontsize=8, fontweight='bold', 
                style='italic', color='#D32F2F',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFEBEE', alpha=0.95, 
                         edgecolor='#D32F2F', linewidth=2.5))
    
    # Input/Output
    ax_flow.text(5, 1.5, 'INPUT: 8 documents (198-312 words each)  →  OUTPUT: 384-word summary',
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', alpha=0.95, 
                         edgecolor='#1B5E20', linewidth=2))
    
    # ===== PANEL 4: LAYER-BY-LAYER COMPARISON (Bottom-Middle) =====
    print("[PANEL 4] Creating layer-by-layer comparison...")
    ax_layers = fig.add_subplot(gs[1, 1])
    ax_layers.set_facecolor('#f5f5f5')
    ax_layers.axis('off')
    
    ax_layers.text(0.5, 0.98, 'LAYER DETAILS', fontsize=12, fontweight='bold', ha='center',
                  transform=ax_layers.transAxes,
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#E0E0E0', alpha=0.95, 
                           edgecolor='#333333', linewidth=2.5))
    
    layer_info = [
        ('Layer 1: Input', '8 documents\nImportance: 0.76-0.95'),
        ('Layer 2: Features', '5 feature types\nTF-IDF, Position, Keywords'),
        ('Layer 3: Clusters', '4 temporal events\nGrouped documents'),
        ('Layer 4: Importance', '8 importance nodes\nScores: 0.68-0.96 ★'),
        ('Layer 5: Selection', '12 sentences selected\nBased on importance'),
        ('Layer 6: Output', 'Final 384-word summary\nAggregated content'),
    ]
    
    colors_layers = ['#BBDEFB', '#90CAF9', '#FFF9C4', '#C8E6C9', '#F8BBD0', '#E1BEE7']
    
    y_pos = 0.88
    for (title, desc), color in zip(layer_info, colors_layers):
        # Layer box
        rect = FancyBboxPatch((0.05, y_pos-0.12), 0.9, 0.11, 
                             boxstyle="round,pad=0.01", 
                             transform=ax_layers.transAxes,
                             edgecolor='#333333', facecolor=color, linewidth=1.5, alpha=0.8)
        ax_layers.add_patch(rect)
        
        ax_layers.text(0.08, y_pos-0.03, title, transform=ax_layers.transAxes,
                      fontsize=8, fontweight='bold', va='center')
        ax_layers.text(0.55, y_pos-0.03, desc, transform=ax_layers.transAxes,
                      fontsize=7, va='center', style='italic')
        
        y_pos -= 0.16
    
    # ===== PANEL 5: QUALITY METRICS RADAR (Bottom-Right) =====
    print("[PANEL 5] Creating evaluation metrics visualization...")
    ax_metrics = fig.add_subplot(gs[1, 2])
    ax_metrics.set_facecolor('#f0f7f4')
    
    ax_metrics.set_title('EVALUATION METRICS', fontsize=11, fontweight='bold', pad=10)
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    
    # Metric data
    metrics_names = ['ROUGE-L', 'BERTScore', 'METEOR', 'Relevance', 'Coherence', 'Factuality']
    metrics_values = [0.67, 0.88, 0.71, 0.85, 0.82, 0.79]
    metric_colors_bar = cm.Set3(np.linspace(0, 1, len(metrics_names)))
    
    # Horizontal bar chart
    y_positions = np.arange(len(metrics_names))
    bars_metrics = ax_metrics.barh(y_positions, metrics_values, 
                                  color=metric_colors_bar, edgecolor='#333333', linewidth=2, alpha=0.85)
    
    ax_metrics.set_yticks(y_positions)
    ax_metrics.set_yticklabels(metrics_names, fontsize=8, fontweight='bold')
    ax_metrics.set_xlabel('Score', fontsize=9, fontweight='bold')
    ax_metrics.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_metrics, metrics_values)):
        ax_metrics.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{val:.2f}', va='center', fontsize=8, fontweight='bold')
    
    # Add threshold line
    ax_metrics.axvline(x=0.75, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Good (0.75)')
    ax_metrics.axvline(x=0.60, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Fair (0.60)')
    
    ax_metrics.legend(fontsize=7, loc='lower right')
    ax_metrics.grid(True, alpha=0.3, axis='x')
    
    # ===== MAIN TITLE & ANNOTATIONS =====
    print("[TITLE] Adding main title and annotations...")
    
    fig.text(0.5, 0.98, 'Comprehensive 3D Multi-View: Importance-Aware Multi-Document Summarization',
            ha='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#E3F2FD', alpha=0.98, 
                     edgecolor='#1976D2', linewidth=3))
    
    fig.text(0.5, 0.945, 'Six-Layer Architecture: Input → Features → Clustering → Importance → Selection → Summary',
            ha='center', fontsize=12, style='italic', color='#333333')
    
    # Bottom annotations
    annotation_text = ('Left Panel: Integrated 3D view of 4-layer pipeline | Top-Right: Importance scoring mechanism (YOUR INNOVATION) |\n'
                      'Bottom-Left: Complete data flow | Bottom-Middle: Layer specifications | Bottom-Right: Quality evaluation metrics')
    fig.text(0.5, 0.01, annotation_text, ha='center', fontsize=9, color='#555555',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', alpha=0.95, 
                     edgecolor='#999999', linewidth=1.5))
    
    # ===== LEGEND =====
    print("[LEGEND] Adding legend...")
    
    legend_elements = [
        mpatches.Circle((0, 0), 0.1, facecolor='#BBDEFB', edgecolor='#1F4E78', linewidth=2, 
                       label='Layer 1: Input Documents (8 sources)'),
        mpatches.Circle((0, 0), 0.1, facecolor='#FFB74D', edgecolor='#C65911', linewidth=2, 
                       label='Layer 2: Event Clusters (4 events)'),
        mpatches.Circle((0, 0), 0.1, facecolor='#81C784', edgecolor='#1B5E20', linewidth=2, 
                       label='Layer 3: Importance Weights (8 nodes) ★ KEY INNOVATION'),
        mpatches.Circle((0, 0), 0.1, facecolor='#66BB6A', edgecolor='#1B5E20', linewidth=2, 
                       label='Layer 4: Final Summary Output'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              framealpha=0.95, edgecolor='black', fancybox=True, shadow=True, 
              bbox_to_anchor=(0.02, 0.30))
    
    # ===== SAVE =====
    print("\n[SAVE] Saving publication-ready figure...")
    
    plt.tight_layout(rect=[0.12, 0.04, 1, 0.94])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_publication_ready_multiview.png',
               dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none',
               pad_inches=0.5)
    
    print("=" * 100)
    print("✓ Publication-ready multi-view figure saved: 3d_publication_ready_multiview.png")
    print("✓ Resolution: 400 DPI (~11200x7200 pixels)")
    print("✓ Features:")
    print("  - Panel 1: Main 3D overview (4-layer pipeline)")
    print("  - Panel 2: Importance weighting detail (YOUR INNOVATION highlighted)")
    print("  - Panel 3: Linear pipeline flow diagram")
    print("  - Panel 4: Layer-by-layer specifications")
    print("  - Panel 5: Quality evaluation metrics")
    print("✓ Professional layout perfect for top-tier conferences")
    print("=" * 100)
    
    plt.show()

if __name__ == "__main__":
    create_publication_ready_figure()
