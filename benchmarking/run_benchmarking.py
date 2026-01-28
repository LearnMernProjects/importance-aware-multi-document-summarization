"""
Main benchmarking orchestration script.
Trains all models and generates comprehensive comparison results.
"""

import logging
import torch
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from config import (
    TRAINING_CONFIG, DATASET_CONFIG, RESULTS_ROOT, CHECKPOINT_ROOT,
    MODELS_TO_BENCHMARK
)
from data.dataset import prepare_dataset
from evaluation.metrics import EvaluationEngine
from training.trainer import (
    PEGASUSTrainer, LEDTrainer, AIMSTrainer, UnifiedTrainer
)
from models.pegasus_model import PEGASUSSummarizer
from models.led_model import LEDSummarizer
from models.aims_model import AIMSSummarizer
from utils.utils import setup_logging, set_seed

logger = setup_logging(__name__)


class BenchmarkingOrchestrator:
    """Orchestrate training and evaluation of all models."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.evaluator = EvaluationEngine(device=device)
        
        self.train_clusters = None
        self.val_clusters = None
        self.test_clusters = None
        
        self.trained_models = {}
        self.results = {
            "per_sample": {},  # {model_id: {metric: [values]}}
            "aggregate": {},   # {model_id: {metric: value}}
        }
        
        logger.info(f"Benchmarking orchestrator initialized (device={device})")
    
    def prepare_data(self) -> None:
        """Load and prepare dataset."""
        logger.info("="*80)
        logger.info("STEP 1: PREPARING DATA")
        logger.info("="*80)
        
        # Check if already processed
        if DATASET_CONFIG["test_clusters_path"].exists():
            logger.info("Loading pre-processed clusters...")
            import json
            with open(DATASET_CONFIG["train_clusters_path"]) as f:
                self.train_clusters = json.load(f)
            with open(DATASET_CONFIG["val_clusters_path"]) as f:
                self.val_clusters = json.load(f)
            with open(DATASET_CONFIG["test_clusters_path"]) as f:
                self.test_clusters = json.load(f)
        else:
            logger.info("Processing clusters from raw data...")
            self.train_clusters, self.val_clusters, self.test_clusters = prepare_dataset()
        
        logger.info(f"Train: {len(self.train_clusters)}, Val: {len(self.val_clusters)}, Test: {len(self.test_clusters)}")
    
    def train_model(self, model_id: str, trainer_class) -> bool:
        """Train a single model."""
        logger.info("="*80)
        logger.info(f"TRAINING: {model_id}")
        logger.info("="*80)
        
        try:
            # Instantiate model
            if model_id == "pegasus":
                model = PEGASUSSummarizer(device=self.device)
                trainer = trainer_class(model, model_id, device=self.device)
            elif model_id == "led":
                model = LEDSummarizer(device=self.device)
                trainer = trainer_class(model, model_id, device=self.device)
            elif model_id == "aims":
                model = AIMSSummarizer(device=self.device)
                trainer = trainer_class(model, model_id, device=self.device)
            else:
                logger.warning(f"Model {model_id} training not yet implemented")
                return False
            
            # Train
            training_result = trainer.train(
                train_clusters=self.train_clusters[:100],  # Subset for demo
                val_clusters=self.val_clusters[:20],
                num_epochs=TRAINING_CONFIG["num_epochs"],
                batch_size=TRAINING_CONFIG["batch_size"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
            )
            
            # Save logs
            trainer.save_training_logs()
            
            self.trained_models[model_id] = model
            
            logger.info(f"✓ Successfully trained {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error training {model_id}: {str(e)}")
            return False
    
    def train_all_models(self) -> None:
        """Train all specified models."""
        logger.info("="*80)
        logger.info("STEP 2: TRAINING ALL MODELS")
        logger.info("="*80)
        
        model_configs = {
            "pegasus": ("pegasus", PEGASUSTrainer),
            "led": ("led", LEDTrainer),
            "aims": ("aims", AIMSTrainer),
        }
        
        for model_id, (config_key, trainer_class) in model_configs.items():
            success = self.train_model(model_id, trainer_class)
            
            if not success:
                logger.warning(f"Skipping evaluation of {model_id}")
    
    def evaluate_on_test_set(self) -> None:
        """Evaluate all trained models on test set."""
        logger.info("="*80)
        logger.info("STEP 3: EVALUATING ON TEST SET")
        logger.info("="*80)
        
        for model_id, model in self.trained_models.items():
            logger.info(f"\nEvaluating {model_id} on {len(self.test_clusters)} test clusters...")
            
            per_sample_metrics = {
                "rouge1": [], "rouge2": [], "rougeL": [],
                "bertscore_f1": [], "redundancy_rate": [],
                "omission_rate": [], "hallucination_rate": [],
                "faithfulness": [], "compression_ratio": [],
            }
            
            pbar = tqdm(total=len(self.test_clusters), desc=f"Evaluating {model_id}")
            
            for cluster in self.test_clusters:
                try:
                    # Get data
                    documents = [a["article_text"] for a in cluster["articles"]]
                    reference = cluster["reference_summary"]
                    
                    # Generate
                    generated = model.generate_summary(documents)
                    
                    # Evaluate
                    metrics = self.evaluator.evaluate_single(documents, reference, generated)
                    
                    for key, value in metrics.items():
                        per_sample_metrics[key].append(value)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating cluster: {str(e)[:100]}")
                
                pbar.update(1)
            
            pbar.close()
            
            # Aggregate
            aggregate_metrics = {}
            for key, values in per_sample_metrics.items():
                if values:
                    aggregate_metrics[f"{key}_mean"] = np.mean(values)
                    aggregate_metrics[f"{key}_std"] = np.std(values)
            
            self.results["per_sample"][model_id] = per_sample_metrics
            self.results["aggregate"][model_id] = aggregate_metrics
            
            logger.info(f"✓ Evaluation complete for {model_id}")
            logger.info(f"  ROUGE-1: {aggregate_metrics.get('rouge1_mean', 0):.4f}")
            logger.info(f"  BERTScore-F1: {aggregate_metrics.get('bertscore_f1_mean', 0):.4f}")
    
    def generate_results(self) -> None:
        """Generate comprehensive result tables."""
        logger.info("="*80)
        logger.info("STEP 4: GENERATING RESULTS")
        logger.info("="*80)
        
        # Main results table
        results_data = []
        for model_id, metrics in self.results["aggregate"].items():
            row = {"model": model_id}
            row.update(metrics)
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_path = RESULTS_ROOT / "results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved: {results_path}")
        
        # Summary table
        summary_metrics = ["rouge1_mean", "rouge2_mean", "rougeL_mean", "bertscore_f1_mean"]
        summary_df = results_df[["model"] + summary_metrics].copy()
        summary_path = RESULTS_ROOT / "summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved: {summary_path}")
        
        # AIMS comparison
        if "aims" in self.results["aggregate"]:
            aims_metrics = self.results["aggregate"]["aims"]
            comparison_data = []
            
            for model_id, metrics in self.results["aggregate"].items():
                if model_id == "aims":
                    continue
                
                row = {"baseline_model": model_id}
                for key, aims_value in aims_metrics.items():
                    metric_name = key.replace("_mean", "")
                    baseline_value = metrics.get(key, 0)
                    
                    improvement = ((aims_value - baseline_value) / abs(baseline_value) * 100
                                  if baseline_value != 0 else 0)
                    
                    row[f"{metric_name}_baseline"] = baseline_value
                    row[f"{metric_name}_aims"] = aims_value
                    row[f"{metric_name}_improvement_%"] = improvement
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = RESULTS_ROOT / "aims_vs_all_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Saved: {comparison_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARKING RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
    
    def run_full_pipeline(self) -> None:
        """Run complete benchmarking pipeline."""
        logger.info("="*80)
        logger.info("STARTING FULL BENCHMARKING PIPELINE")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*80)
        
        # Steps
        self.prepare_data()
        self.train_all_models()
        self.evaluate_on_test_set()
        self.generate_results()
        
        logger.info("="*80)
        logger.info("BENCHMARKING PIPELINE COMPLETE")
        logger.info(f"Results saved to: {RESULTS_ROOT}")
        logger.info("="*80)


def main():
    """Main entry point."""
    set_seed(TRAINING_CONFIG["seed"])
    
    orchestrator = BenchmarkingOrchestrator(device="cuda" if torch.cuda.is_available() else "cpu")
    orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()
