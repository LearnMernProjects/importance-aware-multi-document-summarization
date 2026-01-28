"""
Statistical testing and significance computation for model comparison.
Bootstrap confidence intervals and paired t-tests.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy import stats
import json
from pathlib import Path

from utils.utils import setup_logging

logger = setup_logging(__name__)


class StatisticalTester:
    """Perform statistical significance tests on model results."""
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 10000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level
    
    # ==================== BOOTSTRAP CONFIDENCE INTERVALS ====================
    def bootstrap_ci(
        self,
        values: List[float],
        metric: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            values: Sample values
            metric: "mean", "median", or custom function
        
        Returns:
            {"mean": value, "ci_lower": value, "ci_upper": value, "std": value}
        """
        if not values:
            return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}
        
        values = np.array(values)
        
        # Define statistic function
        if metric == "mean":
            stat_fn = np.mean
        elif metric == "median":
            stat_fn = np.median
        else:
            stat_fn = metric
        
        # Bootstrap samples
        bootstrap_stats = []
        np.random.seed(42)
        
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_stats.append(stat_fn(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute CI
        ci_lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_upper - ci_lower),
        }
    
    # ==================== PAIRED TESTS ====================
    def paired_bootstrap_test(
        self,
        model1_values: List[float],
        model2_values: List[float],
        metric_name: str = "unknown"
    ) -> Dict[str, float]:
        """
        Paired bootstrap test to compare two models.
        Tests null hypothesis: mean(model1) == mean(model2)
        
        Returns:
            {
                "model1_mean": float,
                "model2_mean": float,
                "difference": float,
                "p_value": float,
                "significant": bool,
                "ci_lower": float,
                "ci_upper": float,
            }
        """
        if not model1_values or not model2_values:
            return {
                "model1_mean": 0, "model2_mean": 0, "difference": 0,
                "p_value": 1.0, "significant": False, "ci_lower": 0, "ci_upper": 0
            }
        
        model1_values = np.array(model1_values)
        model2_values = np.array(model2_values)
        
        # Paired differences
        differences = model1_values - model2_values
        obs_diff = np.mean(differences)
        
        # Bootstrap
        bootstrap_diffs = []
        np.random.seed(42)
        
        for _ in range(self.n_bootstrap):
            sample_diff = np.random.choice(differences, size=len(differences), replace=True)
            bootstrap_diffs.append(np.mean(sample_diff))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # P-value: proportion of bootstrap means that cross zero
        p_value = np.mean(bootstrap_diffs >= 0)
        p_value = min(p_value, 1 - p_value) * 2  # Two-tailed
        
        # CI
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha/2) * 100)
        
        significant = p_value < self.alpha
        
        return {
            "metric": metric_name,
            "model1_mean": float(np.mean(model1_values)),
            "model2_mean": float(np.mean(model2_values)),
            "difference": float(obs_diff),
            "p_value": float(p_value),
            "significant": bool(significant),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "effect_size": float(obs_diff / (np.std(differences) + 1e-10)),  # Cohen's d approximation
        }
    
    def paired_ttest(
        self,
        model1_values: List[float],
        model2_values: List[float]
    ) -> Dict:
        """Paired t-test."""
        t_stat, p_value = stats.ttest_rel(model1_values, model2_values)
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
        }
    
    # ==================== MULTIPLE COMPARISONS ====================
    def compute_all_pairwise_comparisons(
        self,
        model_results: Dict[str, List[float]],
        metric_name: str = "metric"
    ) -> List[Dict]:
        """
        Compute pairwise comparisons for all models.
        """
        model_ids = list(model_results.keys())
        comparisons = []
        
        for i, model1_id in enumerate(model_ids):
            for model2_id in model_ids[i+1:]:
                result = self.paired_bootstrap_test(
                    model_results[model1_id],
                    model_results[model2_id],
                    metric_name
                )
                result["model1"] = model1_id
                result["model2"] = model2_id
                comparisons.append(result)
        
        return comparisons
    
    # ==================== RANKING ====================
    def rank_models(
        self,
        model_results: Dict[str, Dict[str, List[float]]],
        metric_name: str
    ) -> List[Tuple[str, float, float, float]]:
        """
        Rank models by metric.
        
        Returns: [(model_id, mean, ci_lower, ci_upper), ...]
        """
        rankings = []
        
        for model_id, metrics in model_results.items():
            if metric_name not in metrics:
                continue
            
            ci = self.bootstrap_ci(metrics[metric_name])
            rankings.append((
                model_id,
                ci["mean"],
                ci["ci_lower"],
                ci["ci_upper"]
            ))
        
        # Sort by mean (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings


class ComparisonSummary:
    """Generate summary comparisons between models."""
    
    @staticmethod
    def create_comparison_table(
        per_sample_results: Dict[str, Dict[str, List[float]]],
        metrics_to_compare: List[str]
    ) -> Dict:
        """
        Create comprehensive comparison table.
        """
        tester = StatisticalTester()
        
        comparison_table = {}
        
        for metric in metrics_to_compare:
            comparison_table[metric] = {
                "rankings": [],
                "pairwise_comparisons": [],
            }
            
            # Get values for this metric
            model_values = {}
            for model_id, metrics_dict in per_sample_results.items():
                if metric in metrics_dict:
                    model_values[model_id] = metrics_dict[metric]
            
            if not model_values:
                continue
            
            # Rankings
            rankings = tester.rank_models(
                {model_id: {metric: vals} for model_id, vals in model_values.items()},
                metric
            )
            comparison_table[metric]["rankings"] = [
                {
                    "rank": rank + 1,
                    "model": model_id,
                    "mean": float(mean),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                }
                for rank, (model_id, mean, ci_lower, ci_upper) in enumerate(rankings)
            ]
            
            # Pairwise comparisons (vs best model)
            if rankings:
                best_model_id = rankings[0][0]
                best_values = model_values[best_model_id]
                
                for model_id, model_vals in model_values.items():
                    if model_id == best_model_id:
                        continue
                    
                    comparison = tester.paired_bootstrap_test(
                        best_values,
                        model_vals,
                        metric
                    )
                    comparison["baseline"] = best_model_id
                    comparison["comparison_model"] = model_id
                    comparison_table[metric]["pairwise_comparisons"].append(comparison)
        
        return comparison_table
    
    @staticmethod
    def aims_vs_all(
        per_sample_results: Dict[str, Dict[str, List[float]]],
        reference_model: str = "aims"
    ) -> Dict:
        """
        Create AIMS vs all other models comparison.
        """
        if reference_model not in per_sample_results:
            return {}
        
        tester = StatisticalTester()
        aims_results = per_sample_results[reference_model]
        
        comparison = {
            "reference_model": reference_model,
            "comparisons": {},
        }
        
        for metric_name, aims_values in aims_results.items():
            comparison["comparisons"][metric_name] = {
                "aims_mean": float(np.mean(aims_values)),
                "aims_std": float(np.std(aims_values)),
                "vs_other_models": [],
            }
            
            for model_id, model_results in per_sample_results.items():
                if model_id == reference_model:
                    continue
                
                if metric_name not in model_results:
                    continue
                
                model_values = model_results[metric_name]
                
                test_result = tester.paired_bootstrap_test(
                    aims_values,
                    model_values,
                    metric_name
                )
                test_result["competitor_model"] = model_id
                test_result["improvement"] = test_result["difference"]
                test_result["improvement_percent"] = (
                    (test_result["difference"] / abs(test_result["model2_mean"]) * 100)
                    if test_result["model2_mean"] != 0 else 0
                )
                
                comparison["comparisons"][metric_name]["vs_other_models"].append(test_result)
        
        return comparison


def generate_statistical_report(
    per_sample_results: Dict[str, Dict[str, List[float]]],
    output_path: Path
) -> None:
    """Generate comprehensive statistical report."""
    logger.info("Generating statistical report...")
    
    metrics_to_analyze = [
        "rouge1", "rouge2", "rougeL", "bertscore_f1",
        "redundancy_rate", "omission_rate", "hallucination_rate",
        "faithfulness", "compression_ratio"
    ]
    
    # Comparison table
    comparison_data = ComparisonSummary.create_comparison_table(
        per_sample_results,
        metrics_to_analyze
    )
    
    # AIMS comparison
    aims_comparison = ComparisonSummary.aims_vs_all(per_sample_results)
    
    report = {
        "timestamp": str(datetime.now().isoformat()),
        "comparison_table": comparison_data,
        "aims_vs_all": aims_comparison,
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Statistical report saved: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*80)
    
    for metric, data in comparison_data.items():
        print(f"\n{metric.upper()}:")
        for item in data["rankings"][:3]:
            print(f"  Rank {item['rank']}: {item['model']} = {item['mean']:.4f} [{item['ci_lower']:.4f}, {item['ci_upper']:.4f}]")


from datetime import datetime

if __name__ == "__main__":
    pass
