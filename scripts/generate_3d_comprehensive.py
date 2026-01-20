import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patches as mpatches
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# High-quality rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.antialiased'] = True

def draw_cylinder_3d(ax, x, y, z, radius, height, color, alpha=0.8, label=''):
    """Draw a 3D cylinder"""
    theta = np.linspace(0, 2*np.pi, 30)
    
    # Top circle
    x_top = x + radius * np.cos(theta)
    y_top = y + radius * np.sin(theta)
    z_top = z + height * np.ones_like(theta)
    
    # Bottom circle
    x_bot = x + radius * np.cos(theta)
    y_bot = y + radius * np.sin(theta)
    z_bot = z * np.ones_like(theta)
    
    # Draw circles
    ax.plot(x_top, y_top, z_top, color=color, linewidth=2)
    ax.plot(x_bot, y_bot, z_bot, color=color, linewidth=2)
    
    # Draw vertical lines
    for i in range(0, len(theta), 3):
        ax.plot([x_top[i], x_bot[i]], [y_top[i], y_bot[i]], [z_top[i], z_bot[i]], 
               color=color, linewidth=1.5, alpha=alpha)
    
    # Fill the top
    for i in range(len(theta)-1):
        vertices = [[x_top[i], y_top[i], z_top[i]], 
                   [x_top[i+1], y_top[i+1], z_top[i+1]],
                   [x, y, z_top[0]]]
        poly = Poly3DCollection([vertices], alpha=alpha*0.6, facecolor=color, edgecolor='none')
        ax.add_collection3d(poly)
    
    if label:
        ax.text(x, y, z + height + 0.5, label, ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))

def draw_sphere_3d(ax, x, y, z, radius, color, alpha=0.8, label='', size_label=''):
    """Draw a 3D sphere"""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=alpha, linewidth=0, shade=True)
    
    if label:
        ax.text(x, y, z + radius + 0.8, label, ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
    
    if size_label:
        ax.text(x, y, z, size_label, ha='center', fontsize=7, fontweight='bold', color='white')

def draw_cone_3d(ax, x, y, z_base, height, radius, color, alpha=0.8, label=''):
    """Draw a 3D cone"""
    theta = np.linspace(0, 2*np.pi, 25)
    
    # Base circle
    x_base = x + radius * np.cos(theta)
    y_base = y + radius * np.sin(theta)
    z_base_line = z_base * np.ones_like(theta)
    
    # Draw base
    ax.plot(x_base, y_base, z_base_line, color=color, linewidth=2)
    
    # Draw lines from base to apex
    for i in range(0, len(theta), 2):
        ax.plot([x_base[i], x], [y_base[i], y], [z_base_line[i], z_base + height], 
               color=color, linewidth=1.5, alpha=alpha)
    
    # Cone surface
    for i in range(len(theta)-1):
        vertices = [[x_base[i], y_base[i], z_base_line[i]], 
                   [x_base[i+1], y_base[i+1], z_base_line[i+1]],
                   [x, y, z_base + height]]
        poly = Poly3DCollection([vertices], alpha=alpha*0.7, facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_collection3d(poly)
    
    if label:
        ax.text(x, y, z_base + height + 0.7, label, ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))

def draw_arrow_3d(ax, start, end, color='black', width=0.1, label=''):
    """Draw 3D arrow"""
    ax.quiver(start[0], start[1], start[2], 
             end[0]-start[0], end[1]-start[1], end[2]-start[2],
             color=color, arrow_length_ratio=0.3, linewidth=width*3, alpha=0.7)
    
    if label:
        mid = [(start[i]+end[i])/2 for i in range(3)]
        ax.text(mid[0], mid[1], mid[2]+0.4, label, ha='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8, edgecolor='black'))

def create_comprehensive_3d_model():
    """
    Create a detailed 3D model explaining the multi-document summarization approach
    """
    fig = plt.figure(figsize=(26, 18))
    fig.patch.set_facecolor('#ffffff')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f5f7fa')
    
    print("=" * 80)
    print("GENERATING COMPREHENSIVE 3D SUMMARIZATION MODEL")
    print("=" * 80)
    
    # ===== LAYER 1: INPUT DOCUMENTS =====
    print("\n[LAYER 1] Rendering Input Documents...")
    doc_radius = 5.5
    num_docs = 8
    angles_docs = np.linspace(0, 2*np.pi, num_docs, endpoint=False)
    
    doc_colors_map = cm.Blues(np.linspace(0.4, 0.95, num_docs))
    doc_importance = np.array([0.95, 0.88, 0.76, 0.82, 0.91, 0.79, 0.85, 0.83])
    doc_sizes = np.array([285, 312, 198, 267, 301, 221, 289, 276])
    
    input_layer_z = 0
    
    for i in range(num_docs):
        x = doc_radius * np.cos(angles_docs[i])
        y = doc_radius * np.sin(angles_docs[i])
        
        # Scale sphere size by importance
        sphere_size = 0.35 + doc_importance[i] * 0.4
        
        draw_sphere_3d(ax, x, y, input_layer_z, sphere_size, 
                      color=doc_colors_map[i], alpha=0.85,
                      label=f'Doc {i+1}', size_label=f'{doc_sizes[i]}w')
        
        # Add importance bar below
        ax.plot([x-0.2, x+0.2], [y, y], [input_layer_z-1.2, input_layer_z-1.2], 
               color=doc_colors_map[i], linewidth=8, alpha=0.7)
        ax.text(x+0.6, y, input_layer_z-1.2, f'{doc_importance[i]:.2f}', 
               fontsize=7, fontweight='bold')
    
    # Label for Input Layer
    ax.text(-7.5, 0, input_layer_z, 'LAYER 1:\nINPUT\nDOCUMENTS\n(8 sources)\n\nImportance:\n0.76-0.95',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#DEEAF6', alpha=0.95, 
                    edgecolor='#1F4E78', linewidth=2.5))
    
    # ===== LAYER 2: FEATURE EXTRACTION =====
    print("[LAYER 2] Rendering Feature Extraction...")
    feature_z = 2.5
    feature_x_positions = np.linspace(-4.5, 4.5, 5)
    
    feature_names = ['TF-IDF', 'Position', 'Keywords', 'Entities', 'Sentiment']
    feature_colors = ['#70AD47', '#FFC000', '#ED7D31', '#A64D79', '#4472C4']
    
    for i, (x, fname, fcolor) in enumerate(zip(feature_x_positions, feature_names, feature_colors)):
        draw_cylinder_3d(ax, x, 0, feature_z, 0.35, 0.8, fcolor, alpha=0.85, label=fname)
    
    # Connections from docs to features
    for i in range(num_docs):
        x_doc = doc_radius * np.cos(angles_docs[i])
        y_doc = doc_radius * np.sin(angles_docs[i])
        
        for j, x_feat in enumerate(feature_x_positions):
            if (i % 5) == j:  # Connect each doc to one feature
                ax.plot([x_doc, x_feat], [y_doc, 0], [input_layer_z, feature_z],
                       color='gray', alpha=0.2, linewidth=1)
    
    ax.text(-7.5, 0, feature_z, 'LAYER 2:\nFEATURE\nEXTRACTION\n\n5 Feature\nTypes',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#F4E5D8', alpha=0.95, 
                    edgecolor='#C65911', linewidth=2.5))
    
    # ===== LAYER 3: EVENT CLUSTERING =====
    print("[LAYER 3] Rendering Event Clustering...")
    cluster_z = 4.8
    num_clusters = 4
    cluster_radius = 3.2
    angles_clusters = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
    
    cluster_colors = cm.Oranges(np.linspace(0.4, 0.95, num_clusters))
    cluster_sizes_data = np.array([3, 2, 2, 1])  # docs per cluster
    cluster_importance_scores = np.array([0.92, 0.81, 0.87, 0.71])
    
    for i in range(num_clusters):
        x_clust = cluster_radius * np.cos(angles_clusters[i])
        y_clust = cluster_radius * np.sin(angles_clusters[i])
        
        # Size represents number of documents
        sphere_size = 0.4 + cluster_sizes_data[i] * 0.25
        
        draw_sphere_3d(ax, x_clust, y_clust, cluster_z, sphere_size,
                      color=cluster_colors[i], alpha=0.85,
                      label=f'Event {i+1}', size_label=f'{cluster_sizes_data[i]}D')
        
        # Importance indicator
        ax.text(x_clust, y_clust-0.8, cluster_z, f'Score:{cluster_importance_scores[i]:.2f}',
               fontsize=7, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
    
    # Connections from features to clusters
    for i, x_feat in enumerate(feature_x_positions):
        target_cluster = i % num_clusters
        x_clust = cluster_radius * np.cos(angles_clusters[target_cluster])
        y_clust = cluster_radius * np.sin(angles_clusters[target_cluster])
        
        ax.plot([x_feat, x_clust], [0, y_clust], [feature_z, cluster_z],
               color='orange', alpha=0.15, linewidth=1)
    
    ax.text(-7.5, 0, cluster_z, 'LAYER 3:\nEVENT\nCLUSTERING\n\n4 Events\nIdentified\n\nTemporal\nGrouping',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFE8D1', alpha=0.95, 
                    edgecolor='#997300', linewidth=2.5))
    
    # ===== LAYER 4: IMPORTANCE WEIGHTING =====
    print("[LAYER 4] Rendering Importance Weighting...")
    weight_z = 7.2
    num_weight_nodes = 8
    weight_angles = np.linspace(0, 2*np.pi, num_weight_nodes, endpoint=False)
    weight_radius = 2.8
    
    importance_scores_full = np.array([0.96, 0.89, 0.78, 0.68, 0.75, 0.84, 0.91, 0.82])
    weight_colors = cm.RdYlGn(importance_scores_full)
    
    for i in range(num_weight_nodes):
        x_weight = weight_radius * np.cos(weight_angles[i])
        y_weight = weight_radius * np.sin(weight_angles[i])
        
        # Size based on importance
        sphere_size = 0.25 + importance_scores_full[i] * 0.35
        
        draw_sphere_3d(ax, x_weight, y_weight, weight_z, sphere_size,
                      color=weight_colors[i], alpha=0.85,
                      label='', size_label=f'{importance_scores_full[i]:.2f}')
    
    # Connections from clusters to weights
    for i in range(num_clusters):
        x_clust = cluster_radius * np.cos(angles_clusters[i])
        y_clust = cluster_radius * np.sin(angles_clusters[i])
        
        for j in range(num_weight_nodes):
            if (i * 2 <= j < i * 2 + 2):
                x_weight = weight_radius * np.cos(weight_angles[j])
                y_weight = weight_radius * np.sin(weight_angles[j])
                
                ax.plot([x_clust, x_weight], [y_clust, y_weight], [cluster_z, weight_z],
                       color='green', alpha=0.15, linewidth=1)
    
    ax.text(-7.5, 0, weight_z, 'LAYER 4:\nIMPORTANCE\nWEIGHTING\n\n8 Nodes\n(0.68-0.96)\n\nSalience\nCalculation',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E5F5E0', alpha=0.95, 
                    edgecolor='#375623', linewidth=2.5))
    
    # ===== LAYER 5: CONTENT SELECTION =====
    print("[LAYER 5] Rendering Content Selection...")
    selection_z = 9.5
    num_selected = 12
    selection_angles = np.linspace(0, 2*np.pi, num_selected, endpoint=False)
    selection_radius = 2.3
    
    selection_scores = np.random.uniform(0.7, 0.95, num_selected)
    selection_colors = cm.viridis(selection_scores)
    
    selected_sentences = []
    for i in range(num_selected):
        x_sel = selection_radius * np.cos(selection_angles[i])
        y_sel = selection_radius * np.sin(selection_angles[i])
        
        # Size based on score
        sphere_size = 0.18 + selection_scores[i] * 0.25
        
        draw_sphere_3d(ax, x_sel, y_sel, selection_z, sphere_size,
                      color=selection_colors[i], alpha=0.8,
                      label='', size_label=f'S{i+1}')
        selected_sentences.append((x_sel, y_sel))
    
    # Connections from weights to selected content
    for i in range(num_weight_nodes):
        x_weight = weight_radius * np.cos(weight_angles[i])
        y_weight = weight_radius * np.sin(weight_angles[i])
        
        # Connect to 2 selected sentences
        for j in range(i*1, min(i*1+2, num_selected)):
            x_sel = selection_radius * np.cos(selection_angles[j])
            y_sel = selection_radius * np.sin(selection_angles[j])
            
            ax.plot([x_weight, x_sel], [y_weight, y_sel], [weight_z, selection_z],
                   color='red', alpha=0.1, linewidth=0.8)
    
    ax.text(-7.5, 0, selection_z, 'LAYER 5:\nCONTENT\nSELECTION\n\n12 Sentences\nSelected\n\nBased on\nImportance',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF4E6', alpha=0.95, 
                    edgecolor='#996600', linewidth=2.5))
    
    # ===== LAYER 6: FINAL SUMMARY =====
    print("[LAYER 6] Rendering Final Summary...")
    summary_z = 11.5
    
    draw_cone_3d(ax, 0, 0, summary_z, 1.2, 1.5, '#70AD47', alpha=0.9, 
                label='FINAL\nSUMMARY')
    
    # Aggregation lines from all selected to summary
    for x_sel, y_sel in selected_sentences:
        ax.plot([x_sel, 0], [y_sel, 0], [selection_z, summary_z],
               color='purple', alpha=0.08, linewidth=0.5)
    
    # Summary statistics box
    summary_stats = 'Output: 384 words\nCoverage: 100%\nRedundancy: 8%\nAvg Score: 0.85'
    ax.text(0, 0, summary_z + 1.8, summary_stats,
           ha='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#C6E0B4', alpha=0.95, 
                    edgecolor='#375623', linewidth=2))
    
    ax.text(-7.5, 0, summary_z, 'LAYER 6:\nFINAL\nSUMMARY\n\nAggregation\n& Output\n\n384 words\n0.85 score',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#C6E0B4', alpha=0.95, 
                    edgecolor='#375623', linewidth=2.5))
    
    # ===== QUALITY METRICS SPHERE =====
    print("[METRICS] Adding Evaluation Metrics...")
    metrics_z = 5.5
    metrics_x = 7.5
    
    metrics_data = {
        'ROUGE-L': 0.67,
        'BERTScore': 0.88,
        'METEOR': 0.71,
        'Relevance': 0.85,
        'Coherence': 0.82,
        'Factuality': 0.79
    }
    
    metric_angles = np.linspace(0, 2*np.pi, len(metrics_data), endpoint=False)
    metric_radius = 1.5
    
    for idx, (metric_name, metric_val) in enumerate(metrics_data.items()):
        x_met = metrics_x + metric_radius * np.cos(metric_angles[idx])
        y_met = metric_radius * np.sin(metric_angles[idx])
        
        metric_color = cm.RdYlGn(metric_val)
        sphere_size = 0.3 + metric_val * 0.25
        
        draw_sphere_3d(ax, x_met, y_met, metrics_z, sphere_size,
                      color=metric_color, alpha=0.8,
                      label='', size_label=f'{metric_val:.2f}')
        
        ax.text(x_met, y_met-0.5, metrics_z, metric_name,
               fontsize=7, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.text(metrics_x, 0, metrics_z - 2.2, 'EVALUATION\nMETRICS\n\n6 Quality\nIndicators',
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0F0F0', alpha=0.95, 
                    edgecolor='#333333', linewidth=2))
    
    # ===== ALGORITHM FLOW BOX =====
    print("[INFO] Adding Algorithm Information...")
    algorithm_text = """
IMPORTANCE-AWARE MULTI-DOCUMENT SUMMARIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INNOVATION: Importance weights guide content selection
• Documents analyzed for relevance and salience
• Temporal events identified and clustered
• Importance scores computed (TF-IDF + Position + Entities)
• Top-scoring sentences selected from each event
• Final summary aggregates all selected content

PIPELINE: Input → Features → Clustering → Importance → Selection → Summary

INPUT: 8 Documents (198-312 words) | OUTPUT: 384-word Summary | QUALITY: 0.85
"""
    
    ax.text(-7, -5.5, 10, algorithm_text,
           ha='left', fontsize=8, fontfamily='monospace', fontweight='bold',
           bbox=dict(boxstyle='round,pad=1', facecolor='#FFF9E6', alpha=0.98, 
                    edgecolor='#333333', linewidth=2.5))
    
    # ===== FORMATTING =====
    print("[FORMAT] Applying formatting...")
    
    ax.set_xlabel('\nDocument & Event Space →', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('\nCluster Distribution →', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_zlabel('\nProcessing Depth (Layers) →', fontsize=13, fontweight='bold', labelpad=15)
    
    ax.set_xlim(-9, 9)
    ax.set_ylim(-6.5, 5.5)
    ax.set_zlim(-2, 13)
    
    # Professional grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)
    
    # Optimal viewing angle
    ax.view_init(elev=20, azim=45)
    
    # ===== TITLE & SUBTITLE =====
    print("[TITLE] Adding title and subtitle...")
    
    title_main = 'Comprehensive 3D Model: Importance-Aware Multi-Document Summarization'
    subtitle = 'Six-Layer Architecture with Hierarchical Processing, Feature Extraction, Event Clustering, & Importance Weighting'
    
    fig.text(0.5, 0.99, title_main, ha='center', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#E3F2FD', alpha=0.98, 
                     edgecolor='#1976D2', linewidth=3))
    fig.text(0.5, 0.955, subtitle, ha='center', fontsize=12, style='italic', color='#333333',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#F5F5F5', alpha=0.95, 
                     edgecolor='#666666', linewidth=1.5))
    
    # ===== LEGEND =====
    print("[LEGEND] Adding legend...")
    
    legend_elements = [
        mpatches.Patch(facecolor='#4472C4', edgecolor='#1F4E78', linewidth=2, 
                      label='Layer 1: Input Documents (8 sources, importance 0.76-0.95)'),
        mpatches.Patch(facecolor='#70AD47', edgecolor='#375623', linewidth=2, 
                      label='Layer 2: Feature Extraction (TF-IDF, Position, Keywords, Entities, Sentiment)'),
        mpatches.Patch(facecolor='#FFC000', edgecolor='#997300', linewidth=2, 
                      label='Layer 3: Event Clustering (4 events identified, temporal grouping)'),
        mpatches.Patch(facecolor='#ED7D31', edgecolor='#C65911', linewidth=2, 
                      label='Layer 4: Importance Weighting (8 nodes, scores 0.68-0.96)'),
        mpatches.Patch(facecolor='#A64D79', edgecolor='#70567B', linewidth=2, 
                      label='Layer 5: Content Selection (12 sentences selected based on importance)'),
        mpatches.Patch(facecolor='#70AD47', edgecolor='#375623', linewidth=2, 
                      label='Layer 6: Final Summary (384 words, aggregated output)'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=9.5, 
             framealpha=0.97, edgecolor='black', fancybox=True, shadow=True, ncol=1)
    legend.get_frame().set_linewidth(2.5)
    
    # ===== SAVE =====
    print("\n[SAVE] Saving high-resolution image...")
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_comprehensive_model.png', 
               dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none',
               pad_inches=0.5)
    
    print("=" * 80)
    print("✓ Comprehensive 3D model saved: 3d_comprehensive_model.png")
    print("✓ Resolution: 400 DPI (Professional Print Quality)")
    print("✓ Dimensions: ~10000x7200 pixels")
    print("✓ Features: 6 distinct layers, 50+ components, full algorithm visualization")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    create_comprehensive_3d_model()
