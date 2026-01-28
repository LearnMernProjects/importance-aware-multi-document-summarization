#!/usr/bin/env python3
"""
Initialize and run the full benchmarking pipeline.
This is the main entry point for the complete framework.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add benchmarking to path
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAINING_CONFIG
from utils.utils import setup_logging, set_seed
from run_benchmarking import BenchmarkingOrchestrator

logger = setup_logging(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Document Summarization Benchmarking Pipeline"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data_only", "train_only", "eval_only"],
        default="full",
        help="Pipeline mode: full (default), data_only, train_only, eval_only"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use: cuda (default) or cpu"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pegasus", "led", "aims"],
        help="Models to train (default: pegasus led aims)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("MULTI-DOCUMENT SUMMARIZATION BENCHMARKING PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*80 + "\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Update config if needed
    TRAINING_CONFIG["num_epochs"] = args.num_epochs
    TRAINING_CONFIG["batch_size"] = args.batch_size
    TRAINING_CONFIG["seed"] = args.seed
    
    # Initialize orchestrator
    orchestrator = BenchmarkingOrchestrator(device=args.device)
    
    try:
        # Run pipeline
        if args.mode == "full":
            orchestrator.run_full_pipeline()
        elif args.mode == "data_only":
            logger.info("Running data preparation only...")
            orchestrator.prepare_data()
        elif args.mode == "train_only":
            logger.info("Running training only...")
            orchestrator.prepare_data()
            orchestrator.train_all_models()
        elif args.mode == "eval_only":
            logger.info("Running evaluation only...")
            orchestrator.prepare_data()
            # Load pre-trained models and evaluate
            orchestrator.evaluate_on_test_set()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nResults saved to: benchmarking/results/")
        print("\nNext steps:")
        print("  1. Review results.csv for comprehensive metrics")
        print("  2. Check aims_vs_all_comparison.csv for AIMS performance")
        print("  3. View plots in results/plots/ for visualizations")
        print("  4. Read statistical_report.json for significance tests")
        print("="*80 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print("\n" + "="*80)
        print("PIPELINE FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("\nCheck benchmarking.log for details")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
