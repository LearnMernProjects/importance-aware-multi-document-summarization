"""
Visualization module for generating publication-ready figures and tables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from utils.utils import setup_logging

logger = setup_logging(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BenchmarkingVisualizer:
    """Create publication-ready visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison_barplot(
        self,
        results_df: pd.DataFrame,
        metric: str = "bertscore_f1_mean",
        figsize: Tuple[int, int] = (12, 6)
    ) -> Path:
        """Create barplot comparing models on a single metric."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#ffffff')
        
        # Sort by metric
        df_sorted = results_df.sort_values(metric, ascending=False)
        
        colors = ['#FF6B6B' if m == 'aims' else '#4ECDC4' for m in df_sorted['model']]
        
        bars = ax.bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = self.output_dir / f"comparison_{metric}.png"
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def plot_metric_heatmap(
        self,
        results_df: pd.DataFrame,
        metrics: List[str],
        figsize: Tuple[int, int] = (12, 8)
    ) -> Path:
        """Create heatmap of all metrics across models."""
        # Extract metrics
        heatmap_data = results_df[["model"] + metrics].set_index("model")
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#ffffff')
        
        sns.heatmap(
            heatmap_data.T,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            center=0.5,
            ax=ax,
            cbar_kws={'label': 'Score'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_title('Multi-Model Evaluation Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_heatmap.png"
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def plot_aims_improvement(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        figsize: Tuple[int, int] = (14, 8)
    ) -> Path:
        """Plot AIMS improvement over baseline models."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.patch.set_facecolor('#ffffff')
        
        for idx, metric in enumerate(metrics[:4]):
            ax = axes[idx // 2, idx % 2]
            
            improvement_col = f"{metric}_improvement_%"
            if improvement_col not in comparison_df.columns:
                continue
            
            improvements = comparison_df.sort_values(improvement_col, ascending=True)
            
            colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in improvements[improvement_col]]
            
            ax.barh(
                range(len(improvements)),
                improvements[improvement_col],
                color=colors,
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )
            
            ax.set_yticks(range(len(improvements)))
            ax.set_yticklabels(improvements['baseline_model'], fontsize=10, fontweight='bold')
            ax.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'AIMS vs Baselines: {metric.title()}', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (idx_val, row) in enumerate(improvements.iterrows()):
                ax.text(
                    row[improvement_col] + 0.5,
                    i,
                    f'{row[improvement_col]:.1f}%',
                    va='center', fontsize=9, fontweight='bold'
                )
        
        plt.suptitle('AIMS Improvement over Baseline Models', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        output_path = self.output_dir / "aims_improvement.png"
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def plot_distribution_comparison(
        self,
        per_sample_results: Dict[str, Dict[str, List[float]]],
        metric: str = "bertscore_f1",
        figsize: Tuple[int, int] = (12, 8)
    ) -> Path:
        """Plot distribution of metric across models."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#ffffff')
        
        # Prepare data
        data_to_plot = []
        labels = []
        
        for model_id in sorted(per_sample_results.keys()):
            if metric in per_sample_results[model_id]:
                data_to_plot.append(per_sample_results[model_id][metric])
                labels.append(model_id.upper() if model_id == "aims" else model_id)
        
        # Create boxplot
        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            widths=0.6,
            showmeans=True
        )
        
        # Color boxes
        colors = ['#FF6B6B' if label == 'AIMS' else '#4ECDC4' for label in labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(f'{metric.title()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of {metric.title()} Across Models', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / f"distribution_{metric}.png"
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def plot_radar_chart(
        self,
        results_df: pd.DataFrame,
        metrics: List[str],
        figsize: Tuple[int, int] = (10, 10)
    ) -> Path:
        """Create radar chart for multi-metric comparison."""
        # Normalize metrics to 0-1
        normalized_df = results_df.copy()
        for metric in metrics:
            if metric in normalized_df.columns:
                min_val = normalized_df[metric].min()
                max_val = normalized_df[metric].max()
                if max_val > min_val:
                    normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('#ffffff')
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors_map = {'aims': '#FF6B6B'}
        
        for idx, row in normalized_df.iterrows():
            values = [row[m] for m in metrics] + [row[metrics[0]]]
            color = colors_map.get(row['model'], f'C{idx}')
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
        
        plt.title('Multi-Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = self.output_dir / "radar_chart.png"
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path


def generate_all_visualizations(
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    per_sample_results: Dict[str, Dict[str, List[float]]],
    output_dir: Path
) -> None:
    """Generate all visualizations."""
    logger.info("="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    visualizer = BenchmarkingVisualizer(output_dir)
    
    # Key metrics
    key_metrics = ["rouge1_mean", "rouge2_mean", "rougeL_mean", "bertscore_f1_mean"]
    
    # Generate plots
    for metric in key_metrics:
        if metric in results_df.columns:
            visualizer.plot_model_comparison_barplot(results_df, metric)
    
    visualizer.plot_metric_heatmap(results_df, key_metrics)
    visualizer.plot_aims_improvement(comparison_df, ["rouge1", "rouge2", "rougeL", "bertscore_f1"])
    visualizer.plot_distribution_comparison(per_sample_results, "bertscore_f1")
    visualizer.plot_radar_chart(results_df, key_metrics)
    
    logger.info("✓ All visualizations generated")


from typing import Tuple

if __name__ == "__main__":
    pass
