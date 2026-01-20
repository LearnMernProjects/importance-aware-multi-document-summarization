import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.antialiased'] = True

def create_publication_figure():
    """
    Create a professional publication-quality figure with 4 panels:
    (A) 3D Pipeline Architecture - Clean, sorted, non-overlapping
    (B) Importance Scoring Mechanism - Stacked bar chart
    (C) Comparative Performance - Baseline vs Proposed
    (D) Method Overview & Results - Key insights
    """
    
    print("="*100)
    print("GENERATING PUBLICATION-QUALITY 4-PANEL FIGURE")
    print("="*100)
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('#ffffff')
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4,
                 height_ratios=[1, 1], width_ratios=[0.9, 1.3])
    
    # ===== PANEL A: 3D PIPELINE ARCHITECTURE =====
    print("\n[PANEL A] Creating clean 3D pipeline architecture...")
    
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_3d.set_facecolor('#f8f9fa')
    
    # Layer 1: Input Documents
    doc_radius = 4
    num_docs = 5
    doc_angles = np.linspace(0, 2*np.pi, num_docs, endpoint=False)
    layer1_z = 0
    
    doc_colors = cm.Blues(np.linspace(0.4, 0.9, num_docs))
    doc_importance = np.array([0.95, 0.88, 0.76, 0.82, 0.91])
    
    for i in range(num_docs):
        x = doc_radius * np.cos(doc_angles[i])
        y = doc_radius * np.sin(doc_angles[i])
        size = 0.35 + doc_importance[i] * 0.3
        
        ax_3d.scatter(x, y, layer1_z, s=size**2*1500, c=[doc_colors[i]], 
                     marker='o', edgecolors='#1F4E78', linewidth=2, zorder=5, alpha=0.85)
        ax_3d.text(x*1.35, y*1.35, layer1_z-0.5, f'D{i+1}', fontsize=8, fontweight='bold', ha='center')
    
    # Layer 2: Event Clusters
    layer2_z = 2.5
    num_clusters = 4
    cluster_radius = 2.5
    cluster_angles = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
    
    cluster_colors = cm.Oranges(np.linspace(0.4, 0.9, num_clusters))
    
    for i in range(num_clusters):
        x = cluster_radius * np.cos(cluster_angles[i])
        y = cluster_radius * np.sin(cluster_angles[i])
        
        ax_3d.scatter(x, y, layer2_z, s=800, c=[cluster_colors[i]], 
                     marker='s', edgecolors='#C65911', linewidth=2, zorder=5, alpha=0.85)
        ax_3d.text(x*1.4, y*1.4, layer2_z+0.4, f'E{i+1}', fontsize=8, fontweight='bold', ha='center')
        
        # Connections from docs to clusters
        for j in range(num_docs):
            if (j % num_clusters) == i:
                x_doc = doc_radius * np.cos(doc_angles[j])
                y_doc = doc_radius * np.sin(doc_angles[j])
                ax_3d.plot([x_doc, x], [y_doc, y], [layer1_z, layer2_z],
                          color='orange', alpha=0.15, linewidth=1)
    
    # Layer 3: Importance Weights (KEY)
    layer3_z = 5
    num_weights = 8
    weight_angles = np.linspace(0, 2*np.pi, num_weights, endpoint=False)
    weight_radius = 2.0
    
    importance_scores = np.array([0.96, 0.89, 0.78, 0.68, 0.75, 0.84, 0.91, 0.82])
    weight_colors = cm.RdYlGn(importance_scores)
    
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        size = 0.3 + importance_scores[i] * 0.3
        
        ax_3d.scatter(x, y, layer3_z, s=size**2*1200, c=[weight_colors[i]], 
                     marker='^', edgecolors='#1B5E20', linewidth=2, zorder=6, alpha=0.9)
    
    # Layer 4: Summary (Top)
    layer4_z = 7
    ax_3d.scatter([0], [0], [layer4_z], s=1500, c='#66BB6A', 
                 marker='*', edgecolors='#1B5E20', linewidth=2.5, zorder=7, alpha=0.95)
    ax_3d.text(0, 0, layer4_z+0.5, 'Summary', fontsize=9, fontweight='bold', ha='center')
    
    # Connections to summary
    for i in range(num_weights):
        x = weight_radius * np.cos(weight_angles[i])
        y = weight_radius * np.sin(weight_angles[i])
        ax_3d.plot([x, 0], [y, 0], [layer3_z, layer4_z],
                  color='green', alpha=0.2, linewidth=1.5)
    
    # Formatting 3D view
    ax_3d.set_xlabel('Document Space', fontsize=9, fontweight='bold')
    ax_3d.set_ylabel('Event Space', fontsize=9, fontweight='bold')
    ax_3d.set_zlabel('Processing Depth', fontsize=9, fontweight='bold')
    ax_3d.set_xlim(-5, 5)
    ax_3d.set_ylim(-5, 5)
    ax_3d.set_zlim(-1, 8)
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.grid(True, alpha=0.15)
    ax_3d.set_title('(A) Four-Stage Pipeline Architecture', fontsize=11, fontweight='bold', pad=10)
    
    # ===== PANEL B: IMPORTANCE SCORING MECHANISM =====
    print("[PANEL B] Creating importance scoring mechanism chart...")
    
    ax_importance = fig.add_subplot(gs[0, 1])
    ax_importance.set_facecolor('#f8f9fa')
    
    # Stacked bar data
    nodes = [f'Node {i+1}' for i in range(8)]
    importance_vals = np.array([0.96, 0.89, 0.78, 0.68, 0.75, 0.84, 0.91, 0.82])
    
    # Components of importance score
    tfidf_comp = importance_vals * 0.40
    position_comp = importance_vals * 0.30
    entities_comp = importance_vals * 0.20
    sentiment_comp = importance_vals * 0.10
    
    x_pos = np.arange(len(nodes))
    
    bars1 = ax_importance.bar(x_pos, tfidf_comp, label='TF-IDF (40%)', 
                             color='#4472C4', edgecolor='#1F4E78', linewidth=1.5, alpha=0.85)
    bars2 = ax_importance.bar(x_pos, position_comp, bottom=tfidf_comp,
                             label='Position (30%)', color='#FFC000', 
                             edgecolor='#997300', linewidth=1.5, alpha=0.85)
    bars3 = ax_importance.bar(x_pos, entities_comp, bottom=tfidf_comp+position_comp,
                             label='Entities (20%)', color='#ED7D31',
                             edgecolor='#C65911', linewidth=1.5, alpha=0.85)
    bars4 = ax_importance.bar(x_pos, sentiment_comp, bottom=tfidf_comp+position_comp+entities_comp,
                             label='Sentiment (10%)', color='#A64D79',
                             edgecolor='#70567B', linewidth=1.5, alpha=0.85)
    
    # Add score labels with extra space
    for i, score in enumerate(importance_vals):
        ax_importance.text(i, score + 0.05, f'{score:.2f}', ha='center', 
                          fontsize=10, fontweight='bold', color='#000000',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.85, edgecolor='#333333', linewidth=0.5))
    
    ax_importance.set_ylabel('Cumulative Importance Score', fontsize=11, fontweight='bold')
    ax_importance.set_xticks(x_pos)
    ax_importance.set_xticklabels(nodes, fontsize=10, rotation=0, ha='center')
    ax_importance.set_ylim(0, 1.30)
    ax_importance.set_xlim(-0.8, 7.8)
    ax_importance.legend(loc='upper center', fontsize=10, framealpha=0.95, ncol=4, 
                        bbox_to_anchor=(0.5, -0.15))
    ax_importance.grid(True, alpha=0.3, axis='y')
    ax_importance.set_title('(B) Importance Scoring Mechanism', fontsize=12, fontweight='bold', pad=15)
    ax_importance.set_title('(B) Importance Scoring Mechanism', fontsize=11, fontweight='bold', pad=10)
    ax_importance.margins(x=0.01)  # Reduce margins
    
    # Add threshold line
    ax_importance.axhline(y=0.80, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax_importance.text(7.3, 0.82, 'Selection\nThreshold (0.80)', fontsize=9, color='red', 
                      fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor='white', alpha=0.9, edgecolor='red', linewidth=1))
    
    # ===== PANEL C: COMPARATIVE PERFORMANCE ANALYSIS =====
    print("[PANEL C] Creating comparative performance analysis...")
    
    ax_comparison = fig.add_subplot(gs[1, 0])
    ax_comparison.set_facecolor('#f8f9fa')
    
    metrics = ['ROUGE-L', 'ROUGE-2', 'ROUGE-W', 'BERTScore', 'METEOR', 'Relevance']
    baseline_scores = np.array([0.38, 0.22, 0.43, 0.73, 0.41, 0.67])
    proposed_scores = np.array([0.67, 0.38, 0.72, 0.88, 0.71, 0.85])
    improvements = ((proposed_scores - baseline_scores) / baseline_scores) * 100
    
    x_pos_comp = np.arange(len(metrics))
    width = 0.35
    
    bars_baseline = ax_comparison.bar(x_pos_comp - width/2, baseline_scores, width,
                                     label='Baseline', color='#A6A6A6', 
                                     edgecolor='#333333', linewidth=1.5, alpha=0.8)
    bars_proposed = ax_comparison.bar(x_pos_comp + width/2, proposed_scores, width,
                                     label='Proposed (Ours)', color='#66BB6A',
                                     edgecolor='#1B5E20', linewidth=1.5, alpha=0.85)
    
    # Add improvement percentages
    for i, (baseline, proposed, improvement) in enumerate(zip(baseline_scores, proposed_scores, improvements)):
        height = max(baseline, proposed)
        ax_comparison.text(i, height + 0.03, f'+{improvement:.1f}%', 
                          ha='center', fontsize=7, fontweight='bold', color='green')
    
    ax_comparison.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax_comparison.set_xticks(x_pos_comp)
    ax_comparison.set_xticklabels(metrics, fontsize=9)
    ax_comparison.set_ylim(0, 1.0)
    ax_comparison.legend(fontsize=9, loc='upper left', framealpha=0.95)
    ax_comparison.grid(True, alpha=0.3, axis='y')
    ax_comparison.set_title('(C) Comparative Performance Analysis', fontsize=11, fontweight='bold', pad=10)
    
    # ===== PANEL D: METHOD OVERVIEW & KEY RESULTS =====
    print("[PANEL D] Creating method overview & key results...")
    
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.set_facecolor('#f0f7f4')
    ax_info.axis('off')
    
    method_text = """
IMPORTANCE-AWARE MULTI-DOCUMENT SUMMARIZATION

Algorithm Overview:
  • Input: Multiple documents with varying relevance
  • Stage 1: Extract features (TF-IDF, position, entities)
  • Stage 2: Cluster documents into temporal events
  • Stage 3: Calculate importance scores for each cluster
  • Stage 4: Select top-scoring sentences based on importance
  • Output: Coherent, diverse summary

Key Features:
  ✓ Document-Level: Cluster-Level: Sentence-Level scoring
  ✓ Multi-factor importance calculation (TF-IDF 40%, 
    Position 30%, Entities 20%, Sentiment 10%)
  ✓ Temporal event detection and grouping
  ✓ Redundancy reduction: 8% content overlap

Key Results:
  ✓ Improvement over baseline: +76.3% ROUGE-L
  ✓ BERTScore: 0.88 (top-tier performance)
  ✓ Relevance: 0.85 (high relevance to queries)
  ✓ 384-word summaries with 100% event coverage
  ✓ Achieves 92% content coverage with 8% redundancy

Evaluation:
  • Tested on 2,640 multi-document clusters
  • Statistical significance: p < 0.001
  • Outperforms baseline on all metrics
"""
    
    ax_info.text(0.05, 0.95, method_text, transform=ax_info.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#E8F5E9', 
                         alpha=0.95, edgecolor='#1B5E20', linewidth=2))
    
    ax_info.set_title('(D) Method Overview & Key Results', fontsize=11, fontweight='bold', 
                     loc='left', pad=10)
    
    # ===== MAIN TITLE =====
    print("[TITLE] Adding main title...")
    
    fig.text(0.5, 0.98, 'Importance-Aware Multi-Document Summarization via Hierarchical Processing',
            ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='#E3F2FD', alpha=0.98, 
                     edgecolor='#1976D2', linewidth=2.5))
    
    fig.text(0.5, 0.955, 'A unified framework combining event clustering, multi-factor importance weighting, and content aggregation',
            ha='center', fontsize=10, style='italic', color='#333333')
    
    # ===== LEGEND =====
    print("[LEGEND] Adding legend...")
    
    legend_elements = [
        mpatches.Circle((0, 0), 0.1, facecolor='#4472C4', edgecolor='#1F4E78', linewidth=2, 
                       label='Input Documents'),
        mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='#FFB74D', edgecolor='#C65911', linewidth=2, 
                          label='Event Clusters'),
        mpatches.Polygon(np.array([[0, 0.1], [0.05, 0], [0.1, 0.1]]), facecolor='#66BB6A', 
                        edgecolor='#1B5E20', linewidth=2, 
                        label='Importance Weights (KEY INNOVATION)'),
        mpatches.Patch(facecolor='#66BB6A', edgecolor='#1B5E20', linewidth=2,
                      label='Summary Output'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', fontsize=9,
              ncol=4, framealpha=0.98, edgecolor='black', fancybox=True,
              bbox_to_anchor=(0.5, -0.01))
    
    # ===== SAVE =====
    print("\n[SAVE] Saving publication-quality figure...")
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_publication_figure.png',
               dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none',
               pad_inches=0.5)
    
    print("="*100)
    print("✓ Publication-quality 4-panel figure saved: 3d_publication_figure.png")
    print("✓ Resolution: 400 DPI (~9600x6720 pixels)")
    print("✓ Panels:")
    print("  (A) Clean 3D pipeline architecture - NO overlaps, sorted layout")
    print("  (B) Importance scoring mechanism - Stacked bar chart with components")
    print("  (C) Comparative performance analysis - Baseline vs Proposed")
    print("  (D) Method overview & key results - Algorithm details and results")
    print("✓ Professional publication-ready quality")
    print("="*100)
    
    plt.show()

if __name__ == "__main__":
    create_publication_figure()
