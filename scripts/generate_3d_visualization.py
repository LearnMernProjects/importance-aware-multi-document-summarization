import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set high-quality rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['text.antialiased'] = True
plt.rcParams['figure.figsize'] = (20, 16)

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs = [x1, x1 + dx]
        ys = [y1, y1 + dy]
        zs = [z1, z1 + dz]
        
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions(((xs[0], ys[0]), (xs[1], ys[1])))
        super().draw(renderer)

def create_3d_summarization_viz():
    """
    Create a high-quality 3D visualization showing multi-document summarization pipeline
    with detailed importance hierarchy and information flow
    """
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#ffffff')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f5f7fa')
    
    # ===== LAYER 1: INPUT DOCUMENTS (Bottom Layer) =====
    print("Rendering Layer 1: Input Documents...")
    num_documents = 10
    doc_angles = np.linspace(0, 2*np.pi, num_documents, endpoint=False)
    doc_radius = 4.2
    
    doc_x = doc_radius * np.cos(doc_angles)
    doc_y = doc_radius * np.sin(doc_angles)
    doc_z = np.zeros(num_documents)
    
    # Document metadata
    doc_words = np.array([245, 312, 198, 267, 289, 221, 334, 276, 298, 251])
    doc_relevance = np.array([0.92, 0.87, 0.76, 0.89, 0.85, 0.79, 0.88, 0.83, 0.91, 0.84])
    
    colors_docs = plt.cm.Blues(np.linspace(0.35, 0.95, num_documents))
    
    for i in range(num_documents):
        ax.scatter(doc_x[i], doc_y[i], doc_z[i], s=1200, c=[colors_docs[i]], 
                  marker='s', edgecolors='#1f77b4', linewidth=2.5, zorder=10, alpha=0.9)
        
        # Multi-line label with metadata
        ax.text(doc_x[i], doc_y[i], doc_z[i]-0.7, f'Doc {i+1}', 
               ha='center', fontsize=9, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#1f77b4'))
        
        # Add word count indicator
        ax.text(doc_x[i]*1.2, doc_y[i]*1.2, doc_z[i]+0.3, 
               f'{doc_words[i]}w\n{doc_relevance[i]:.2f}', 
               ha='center', fontsize=7, style='italic', color='#555555')
    
    # ===== LAYER 2: EVENT CLUSTERING (Middle Layer) =====
    print("Rendering Layer 2: Event Clusters...")
    cluster_z = 3.5
    num_clusters = 5
    cluster_angles = np.linspace(0, 2*np.pi, num_clusters, endpoint=False)
    cluster_radius = 2.8
    
    cluster_x = cluster_radius * np.cos(cluster_angles)
    cluster_y = cluster_radius * np.sin(cluster_angles)
    cluster_z_vals = np.full(num_clusters, cluster_z)
    
    # Cluster metadata
    cluster_sizes = np.array([4, 3, 4, 2, 3])  # documents per cluster
    cluster_importance = np.array([0.94, 0.81, 0.87, 0.73, 0.85])
    
    colors_clusters = plt.cm.Greens(np.linspace(0.35, 0.95, num_clusters))
    
    for i in range(num_clusters):
        size_scale = 800 + cluster_sizes[i] * 300
        ax.scatter(cluster_x[i], cluster_y[i], cluster_z_vals[i], s=size_scale, 
                  c=[colors_clusters[i]], marker='o', edgecolors='#2ca02c', 
                  linewidth=3, zorder=10, alpha=0.85)
        
        # Cluster label
        ax.text(cluster_x[i], cluster_y[i], cluster_z_vals[i]-0.8, f'Event {i+1}', 
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#2ca02c'))
        
        # Cluster details
        ax.text(cluster_x[i]*1.25, cluster_y[i]*1.25, cluster_z_vals[i]+0.4, 
               f'Docs: {cluster_sizes[i]}\nScore: {cluster_importance[i]:.2f}', 
               ha='center', fontsize=8, style='italic', 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.6))
    
    # Draw detailed connections: Documents -> Clusters
    print("Drawing document to cluster connections...")
    for i in range(num_documents):
        # Each document connects to 1-2 clusters with gradient effect
        cluster_assignments = [i % num_clusters, (i + 1) % num_clusters]
        for cluster_idx in cluster_assignments[:1]:  # Primary connection
            connection_strength = doc_relevance[i]
            ax.plot([doc_x[i], cluster_x[cluster_idx]], 
                   [doc_y[i], cluster_y[cluster_idx]], 
                   [doc_z[i], cluster_z_vals[cluster_idx]], 
                   color='#4472C4', alpha=0.35, linewidth=2)
    
    # ===== LAYER 3: IMPORTANCE SCORING (Upper Middle Layer) =====
    print("Rendering Layer 3: Importance Scoring...")
    importance_z = 6.0
    num_importance_nodes = 8
    importance_angles = np.linspace(0, 2*np.pi, num_importance_nodes, endpoint=False)
    importance_radius = 2.2
    
    importance_x = importance_radius * np.cos(importance_angles)
    importance_y = importance_radius * np.sin(importance_angles)
    importance_z_vals = np.full(num_importance_nodes, importance_z)
    
    # Importance scores with detailed metadata
    importance_scores = np.array([0.96, 0.88, 0.79, 0.68, 0.75, 0.84, 0.91, 0.82])
    importance_categories = ['Event 1a', 'Event 1b', 'Event 2a', 'Event 2b', 'Event 3a', 'Event 3b', 'Event 4a', 'Event 4b']
    importance_sizes = 500 + importance_scores * 700
    colors_importance = plt.cm.RdYlGn(importance_scores)
    
    for i in range(num_importance_nodes):
        ax.scatter(importance_x[i], importance_y[i], importance_z_vals[i], 
                  s=importance_sizes[i], c=[colors_importance[i]], marker='^',
                  edgecolors='#d62728', linewidth=2.5, zorder=10, alpha=0.88)
        
        ax.text(importance_x[i], importance_y[i], importance_z_vals[i]-0.7, 
               f'{importance_scores[i]:.2f}', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.75, edgecolor='#d62728'))
        
        # Category label
        ax.text(importance_x[i]*1.3, importance_y[i]*1.3, importance_z_vals[i]+0.3, 
               importance_categories[i], ha='center', fontsize=7, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.6))
    
    # Connections: Clusters -> Importance scores with weighted edges
    print("Drawing cluster to importance connections...")
    for i in range(num_clusters):
        for j in range(num_importance_nodes):
            if (i * 2 <= j < i * 2 + 2):  # Logical connections
                ax.plot([cluster_x[i], importance_x[j]], 
                       [cluster_y[i], importance_y[j]], 
                       [cluster_z_vals[i], importance_z_vals[j]], 
                       color='#70AD47', alpha=0.4, linewidth=2)
    
    # ===== LAYER 4: FINAL SUMMARY (Top Layer) =====
    print("Rendering Layer 4: Summary Output...")
    summary_z = 8.5
    summary_x = 0
    summary_y = 0
    
    # Large star marker for summary
    ax.scatter([summary_x], [summary_y], [summary_z], s=4000, c='#ff7f0e', 
              marker='*', edgecolors='#d62728', linewidth=4, zorder=12, alpha=0.95)
    
    # Summary label with statistics
    summary_stats = "FINAL SUMMARY\n384 words | 0.88 avg score\n5 key events | 100% coverage"
    ax.text(summary_x, summary_y-1.3, summary_z, summary_stats, 
           ha='center', fontsize=11, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#ffe6cc', alpha=0.95, 
                    edgecolor='#d62728', linewidth=2))
    
    # Draw all importance scores to summary with thick connections
    print("Drawing importance to summary connections...")
    for i in range(num_importance_nodes):
        ax.plot([importance_x[i], summary_x], 
               [importance_y[i], summary_y], 
               [importance_z_vals[i], summary_z], 
               color='#ff7f0e', alpha=0.45, linewidth=2.5)
    
    # ===== LAYER 5: EVALUATION METRICS (Outer Ring) =====
    print("Rendering Layer 5: Evaluation Metrics...")
    metrics_z = 4.8
    metrics = ['ROUGE-L\n(0.67)', 'BERTScore\n(0.88)', 'METEOR\n(0.71)', 
               'Relevance\n(0.85)', 'Coherence\n(0.82)', 'Factuality\n(0.79)']
    metric_scores = [0.67, 0.88, 0.71, 0.85, 0.82, 0.79]
    num_metrics = len(metrics)
    metrics_angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    metrics_radius = 5.2
    
    metrics_x = metrics_radius * np.cos(metrics_angles)
    metrics_y = metrics_radius * np.sin(metrics_angles)
    metrics_z_vals = np.full(num_metrics, metrics_z)
    
    colors_metrics = plt.cm.Purples(np.array(metric_scores))
    
    for i in range(num_metrics):
        size_metric = 500 + metric_scores[i] * 400
        ax.scatter(metrics_x[i], metrics_y[i], metrics_z_vals[i], s=size_metric, 
                  c=[colors_metrics[i]], marker='D', edgecolors='#9467bd', 
                  linewidth=2.5, zorder=9, alpha=0.85)
        
        ax.text(metrics_x[i]*1.22, metrics_y[i]*1.22, metrics_z_vals[i]+0.2, 
               metrics[i], ha='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8, edgecolor='#9467bd'))
    
    # ===== LAYER 6: PIPELINE INFO (Background Context) =====
    print("Rendering Layer 6: Pipeline Information...")
    # Add text boxes with layer descriptions
    layer_descriptions = [
        ('Input\nDocuments', -5.5, -5.5, 0.5, '#4472C4'),
        ('Event\nClustering', -5.5, 5.5, 3.5, '#70AD47'),
        ('Importance\nScoring', 5.5, 5.5, 6.0, '#ff7f0e'),
        ('Summary\nOutput', 5.5, -5.5, 8.5, '#d62728'),
    ]
    
    for desc, x, y, z, color in layer_descriptions:
        ax.text(x, y, z, desc, ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.25, 
                        edgecolor=color, linewidth=2))
    
    # ===== FORMATTING & STYLING =====
    print("Applying final formatting...")
    
    ax.set_xlabel('\n→ Document Space (Features & Content)', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('\n→ Event/Cluster Space (Temporal Context)', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_zlabel('\n→ Processing Hierarchy (Pipeline Stages)', fontsize=13, fontweight='bold', labelpad=15)
    
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_zlim(-1.5, 9.5)
    
    # Professional grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Set viewing angle for maximum visual impact and clarity
    ax.view_init(elev=22, azim=50)
    
    # ===== LEGEND WITH DETAILED INFORMATION =====
    legend_elements = [
        mpatches.Patch(facecolor='#4472C4', edgecolor='#1f77b4', linewidth=2, label='Input Documents (word count & relevance)'),
        mpatches.Patch(facecolor='#70AD47', edgecolor='#2ca02c', linewidth=2, label='Event Clusters (size & score)'),
        mpatches.Patch(facecolor='#ff7f0e', edgecolor='#d62728', linewidth=2, label='Importance Scores (0.68-0.96)'),
        mpatches.Patch(facecolor='#d62728', edgecolor='#8B0000', linewidth=2, label='Final Summary Output'),
        mpatches.Patch(facecolor='#9467bd', edgecolor='#9467bd', linewidth=2, label='Evaluation Metrics (ROUGE, BERTScore, etc.)'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(2)
    
    # Add main title with subtitle
    title_text = 'Multi-Document Summarization: 3D Information Flow Architecture\n'
    subtitle_text = 'Importance-Aware Processing Pipeline with Event Clustering & Hierarchical Aggregation'
    
    fig.text(0.5, 0.98, title_text + subtitle_text, ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8', alpha=0.9, edgecolor='#1f77b4', linewidth=2))
    
    # Add footer with metadata
    footer_text = 'Input: 10 Documents | Clusters: 5 Events | Importance Nodes: 8 | Metrics: 6 Evaluation Methods | Output: 384-word Summary'
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, style='italic', color='#555555')
    
    print("Saving high-resolution image...")
    
    # Save with maximum quality settings
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig('c:\\Users\\Viraj Naik\\Desktop\\Suvidha\\3d_summarization_architecture_hd.png', 
               dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none',
               pad_inches=0.3)
    
    print("✓ High-resolution 3D visualization saved as: 3d_summarization_architecture_hd.png")
    print("✓ Resolution: 400 DPI (Professional Print Quality)")
    print("✓ Size: ~8000x6400 pixels")
    print("✓ Format: PNG with transparent background option")
    
    plt.show()

if __name__ == "__main__":
    create_3d_summarization_viz()

if __name__ == "__main__":
    create_3d_summarization_viz()
