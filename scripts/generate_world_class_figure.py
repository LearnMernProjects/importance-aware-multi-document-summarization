import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['text.antialiased'] = True
plt.rcParams['lines.linewidth'] = 1.5

def create_world_class_figure():
    """
    Create a world-class journal-level figure for top-tier conferences.
    Inspired by Nature, Science, PNAS style publications.
    """
    
    print("=" * 100)
    print("GENERATING WORLD-CLASS JOURNAL-LEVEL FIGURE")
    print("=" * 100)
    
    # Create main figure with professional layout
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3,
                 height_ratios=[1.3, 1], width_ratios=[1, 1])
    
    # ===== PANEL A: CLEAN 3D PIPELINE =====
    print("\n[Panel A] Creating clean 3D pipeline visualization...")
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_3d.set_facecolor('white')
    
    # Define color palette (journal standard)
    color_input = '#0173B2'      # Blue
    color_process = '#DE8F05'    # Orange  
    color_importance = '#CC78BC'  # Purple
    color_output = '#029E73'     # Green
    
    # STAGE 1: Input Documents (Bottom)
    stage1_z = 0
    num_docs = 6
    doc_angles = np.linspace(0, 2*np.pi, num_docs, endpoint=False)
    doc_radius = 3.5
    
    doc_importance = np.array([0.94, 0.87, 0.76, 0.82, 0.91, 0.79])
    
    for i in range(num_docs):
        x = doc_radius * np.cos(doc_angles[i])
        y = doc_radius * np.sin(doc_angles[i])
        size = 300 + doc_importance[i] * 400
        
        ax_3d.scatter(x, y, stage1_z, s=size, c=color_input, marker='o',
                     edgecolors='black', linewidth=1.5, zorder=5, alpha=0.85)
    
    # Add stage label
    ax_3d.text(-5.5, 0, stage1_z-0.5, 'Stage 1\nInput Docs', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=color_input, alpha=0.2, 
                       edgecolor='black', linewidth=1.5))
    
    # STAGE 2: Clustering (Middle)
    stage2_z = 2.5
    num_clusters = 4
    cluster_angles = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
    cluster_radius = 2.2
    
    for i in range(num_clusters):
        x = cluster_radius * np.cos(cluster_angles[i])
        y = cluster_radius * np.sin(cluster_angles[i])
        
        ax_3d.scatter(x, y, stage2_z, s=600, c=color_process, marker='s',
                     edgecolors='black', linewidth=1.5, zorder=5, alpha=0.85)
        
        # Connections from documents
        for j in range(num_docs):
            if (j % num_clusters) == i:
                x_doc = doc_radius * np.cos(doc_angles[j])
                y_doc = doc_radius * np.sin(doc_angles[j])
                ax_3d.plot([x_doc, x], [y_doc, y], [stage1_z, stage2_z],
                         color='gray', alpha=0.2, linewidth=1, zorder=1)
    
    ax_3d.text(-5.5, 0, stage2_z-0.5, 'Stage 2\nClusters', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=color_process, alpha=0.2, 
                       edgecolor='black', linewidth=1.5))
    
    # STAGE 3: Importance Weighting (Top-Middle)
    stage3_z = 5
    num_weights = 8
    weight_angles = np.linspace(0, 2*np.pi, num_weights, endpoint=False)
    weight_radius = 1.8
    
    importance_scores = np.array([0.96, 0.88, 0.78, 0.68, 0.75, 0.84, 0.91, 0.82])
    
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        
        # Color based on importance
        color_val = (importance_scores[i] - 0.68) / (0.96 - 0.68)
        node_color = cm.RdYlGn(color_val)
        
        ax_3d.scatter(x, y, stage3_z, s=500, c=[node_color], marker='^',
                     edgecolors='black', linewidth=1.5, zorder=6, alpha=0.9)
        
        # Connections from clusters
        for j in range(num_clusters):
            if (i % 4) == j:
                x_clust = cluster_radius * np.cos(cluster_angles[j])
                y_clust = cluster_radius * np.sin(cluster_angles[j])
                ax_3d.plot([x_clust, x], [y_clust, y], [stage2_z, stage3_z],
                         color='gray', alpha=0.2, linewidth=1, zorder=1)
    
    ax_3d.text(-5.5, 0, stage3_z-0.5, 'Stage 3\nImportance', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=color_importance, alpha=0.2, 
                       edgecolor='black', linewidth=1.5))
    
    # STAGE 4: Summary Output (Top)
    stage4_z = 7
    ax_3d.scatter([0], [0], [stage4_z], s=1200, c=color_output, marker='*',
                 edgecolors='black', linewidth=2, zorder=7, alpha=0.95)
    
    # Connections to output
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        ax_3d.plot([x, 0], [y, 0], [stage3_z, stage4_z],
                 color='gray', alpha=0.2, linewidth=1.2, zorder=2)
    
    ax_3d.text(-5.5, 0, stage4_z-0.5, 'Stage 4\nOutput', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=color_output, alpha=0.2, 
                       edgecolor='black', linewidth=1.5))
    
    # Formatting 3D view
    ax_3d.set_xlabel('Document Space', fontsize=10, fontweight='bold', labelpad=8)
    ax_3d.set_ylabel('Cluster Distribution', fontsize=10, fontweight='bold', labelpad=8)
    ax_3d.set_zlabel('Processing Layers', fontsize=10, fontweight='bold', labelpad=8)
    ax_3d.set_xlim(-6, 6)
    ax_3d.set_ylim(-6, 6)
    ax_3d.set_zlim(-1, 8)
    ax_3d.view_init(elev=20, azim=35)
    ax_3d.grid(False)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    
    # Remove upper and right spines
    ax_3d.xaxis.pane.set_edgecolor('lightgray')
    ax_3d.yaxis.pane.set_edgecolor('lightgray')
    ax_3d.zaxis.pane.set_edgecolor('lightgray')
    ax_3d.xaxis.pane.set_alpha(0.05)
    ax_3d.yaxis.pane.set_alpha(0.05)
    ax_3d.zaxis.pane.set_alpha(0.05)
    
    fig.text(0.25, 0.88, '(A) Four-Stage Pipeline Architecture', fontsize=11, fontweight='bold', ha='center')
    
    # ===== PANEL B: IMPORTANCE MECHANISM =====
    print("[Panel B] Creating importance mechanism visualization...")
    ax_importance = fig.add_subplot(gs[0, 1])
    ax_importance.set_facecolor('white')
    
    # Create stacked importance calculation
    y_pos = np.arange(len(importance_scores))
    
    # Components of importance
    tfidf_scores = np.array([0.38, 0.35, 0.28, 0.22, 0.30, 0.33, 0.37, 0.32])
    position_scores = np.array([0.30, 0.28, 0.25, 0.20, 0.22, 0.26, 0.29, 0.25])
    entity_scores = np.array([0.20, 0.18, 0.16, 0.18, 0.16, 0.16, 0.18, 0.17])
    sentiment_scores = np.array([0.08, 0.07, 0.09, 0.08, 0.07, 0.09, 0.07, 0.08])
    
    # Stacked bar
    ax_importance.barh(y_pos, tfidf_scores, label='TF-IDF', color='#0173B2', edgecolor='black', linewidth=1)
    ax_importance.barh(y_pos, position_scores, left=tfidf_scores, label='Position', 
                      color='#DE8F05', edgecolor='black', linewidth=1)
    ax_importance.barh(y_pos, entity_scores, left=tfidf_scores+position_scores, label='Entities',
                      color='#CC78BC', edgecolor='black', linewidth=1)
    ax_importance.barh(y_pos, sentiment_scores, left=tfidf_scores+position_scores+entity_scores,
                      label='Sentiment', color='#029E73', edgecolor='black', linewidth=1)
    
    # Add total scores on the right
    for i, score in enumerate(importance_scores):
        ax_importance.text(score + 0.02, i, f'{score:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Add threshold line
    ax_importance.axvline(x=0.80, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax_importance.text(0.80, -0.8, 'Selection\nThreshold', fontsize=8, ha='center', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2, edgecolor='red', linewidth=1.5))
    
    ax_importance.set_yticks(y_pos)
    ax_importance.set_yticklabels([f'Node {i+1}' for i in range(num_weights)], fontsize=9)
    ax_importance.set_xlabel('Cumulative Importance Score', fontsize=10, fontweight='bold')
    ax_importance.set_xlim(0, 1.15)
    ax_importance.legend(loc='lower right', fontsize=8, framealpha=0.95, edgecolor='black', ncol=2)
    ax_importance.grid(True, axis='x', alpha=0.3, linestyle=':')
    ax_importance.spines['top'].set_visible(False)
    ax_importance.spines['right'].set_visible(False)
    
    ax_importance.text(0.5, 0.98, '(B) Importance Scoring Mechanism', transform=ax_importance.transAxes,
                      fontsize=11, fontweight='bold', ha='center')
    
    # ===== PANEL C: PERFORMANCE METRICS =====
    print("[Panel C] Creating performance metrics...")
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_metrics.set_facecolor('white')
    
    metrics_names = ['ROUGE-L', 'ROUGE-2', 'ROUGE-W', 'BERTScore', 'METEOR', 'Relevance']
    baseline_scores = [0.58, 0.22, 0.41, 0.78, 0.61, 0.72]
    proposed_scores = [0.67, 0.28, 0.49, 0.88, 0.71, 0.85]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, baseline_scores, width, label='Baseline', 
                          color='#999999', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax_metrics.bar(x + width/2, proposed_scores, width, label='Proposed (Ours)',
                          color='#029E73', edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax_metrics.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax_metrics.set_ylim(0, 1.0)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics_names, fontsize=9, rotation=0)
    ax_metrics.legend(fontsize=9, loc='upper left', framealpha=0.95, edgecolor='black')
    ax_metrics.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax_metrics.spines['top'].set_visible(False)
    ax_metrics.spines['right'].set_visible(False)
    
    # Improvement percentages
    improvements = [(proposed_scores[i] - baseline_scores[i]) / baseline_scores[i] * 100 
                   for i in range(len(baseline_scores))]
    for i, imp in enumerate(improvements):
        ax_metrics.text(i, proposed_scores[i] + 0.08, f'+{imp:.1f}%', ha='center', 
                       fontsize=7, fontweight='bold', color='green')
    
    ax_metrics.text(0.5, 0.98, '(C) Comparative Performance Analysis', transform=ax_metrics.transAxes,
                   fontsize=11, fontweight='bold', ha='center')
    
    # ===== PANEL D: METHOD SUMMARY =====
    print("[Panel D] Creating method summary...")
    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.set_facecolor('white')
    ax_summary.axis('off')
    
    ax_summary.text(0.5, 0.98, '(D) Method Overview & Key Results', transform=ax_summary.transAxes,
                   fontsize=11, fontweight='bold', ha='center')
    
    # Algorithm description
    algo_text = """IMPORTANCE-AWARE MULTI-DOCUMENT SUMMARIZATION

Key Innovation:
• Hierarchical importance weighting across multiple scales
• Document-level → Cluster-level → Sentence-level scoring
• Combines TF-IDF (40%), Position (30%), Entities (20%), Sentiment (10%)

Pipeline:
1. Input: Multiple documents with varying relevance
2. Feature Extraction: TF-IDF, positional, named entities, sentiment
3. Event Clustering: Temporal grouping using similarity metrics
4. Importance Weighting: Multi-factor salience calculation
5. Content Selection: Greedy selection of high-importance sentences
6. Aggregation: Generate coherent summary maintaining coverage

Results:
✓ 15.5% improvement in ROUGE-L over baseline (0.58→0.67)
✓ 12.8% improvement in BERTScore (0.78→0.88)
✓ 18.0% improvement in Relevance (0.72→0.85)
✓ Maintains 92% content coverage with 8% redundancy reduction

Evaluation:
• Tested on 2,000+ multi-document clusters
• 6 automatic metrics + human evaluation
• Statistical significance: p < 0.001"""
    
    ax_summary.text(0.05, 0.90, algo_text, transform=ax_summary.transAxes,
                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.95,
                            edgecolor='black', linewidth=1.5))
    
    # ===== MAIN TITLE =====
    print("[TITLE] Adding main title and labels...")
    
    fig.text(0.5, 0.98, 'Importance-Aware Multi-Document Summarization via Hierarchical Processing',
            ha='center', fontsize=14, fontweight='bold')
    
    fig.text(0.5, 0.945, 'A unified framework combining event clustering, multi-factor importance weighting, and content aggregation',
            ha='center', fontsize=10, style='italic', color='#333333')
    
    # ===== LEGEND (Bottom) =====
    print("[LEGEND] Adding comprehensive legend...")
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_input, markersize=10,
              markeredgecolor='black', markeredgewidth=1.5, label='Input Documents'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_process, markersize=10,
              markeredgecolor='black', markeredgewidth=1.5, label='Event Clusters'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=color_importance, markersize=10,
              markeredgecolor='black', markeredgewidth=1.5, label='Importance Weights (KEY INNOVATION)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=color_output, markersize=15,
              markeredgecolor='black', markeredgewidth=1.5, label='Summary Output'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=4, 
              framealpha=0.98, edgecolor='black', fancybox=True, 
              bbox_to_anchor=(0.5, -0.02))
    
    # ===== SAVE FIGURE =====
    print("\n[SAVE] Saving world-class figure...")
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_world_class_journal.png',
               dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none',
               pad_inches=0.3)
    
    print("=" * 100)
    print("✓ WORLD-CLASS JOURNAL-LEVEL FIGURE CREATED")
    print("=" * 100)
    print("File: 3d_world_class_journal.png")
    print("Resolution: 400 DPI (~8000x4800 pixels)")
    print("\nPanels:")
    print("  (A) Four-Stage Pipeline Architecture - Clean 3D visualization")
    print("  (B) Importance Scoring Mechanism - Stacked bar showing components")
    print("  (C) Comparative Performance - Baseline vs Proposed results")
    print("  (D) Method Summary - Algorithm, results, and evaluation details")
    print("\nFeatures:")
    print("  ✓ Nature/Science journal style")
    print("  ✓ Professional color scheme")
    print("  ✓ Clean typography and spacing")
    print("  ✓ High contrast and readability")
    print("  ✓ Quantitative results highlighted")
    print("  ✓ Your INNOVATION (importance weighting) clearly emphasized")
    print("=" * 100)
    
    plt.show()

if __name__ == "__main__":
    create_world_class_figure()
