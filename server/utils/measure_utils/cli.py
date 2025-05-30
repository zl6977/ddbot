"""
Command-line interface utilities for system evaluation.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.configs import log_config

from .evaluation_runner import run_evaluation

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for command-line interface.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Run system evaluation with configurable parameters.")
    
    # Sample configuration
    parser.add_argument("--sample_size", type=int, default=1, 
                       help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--sample_non_none_only", action='store_true', 
                       help="Sample only mnemonics whose labels are not None")
    parser.add_argument("--sample_seed", type=int, default=42, 
                       help="Random seed for sampling (default: 42)")
    
    # Evaluation parameters
    parser.add_argument("--threshold", type=float, default=0.9, 
                       help="Probability threshold for validation (default: 1.0)")
    
    # AI configuration
    parser.add_argument("--use_chain_of_thought", action='store_true', 
                       help="Use Chain of Thought reasoning")
    parser.add_argument("--dont_interpret", action='store_false', 
                       help="Use interpretation for better results")
    parser.add_argument("--model", type=str, default='gpt-4o-mini', 
                       choices=['gpt-4o-mini', 'gpt-4.1-nano', 'gpt-4.1-mini'], 
                       help="Model to use for evaluation (default: 'gpt-4o-mini')")
    
    # Advanced parameters
    parser.add_argument("--number_of_candidates", type=json.loads, 
                       default='{"Quantity_class": 5, "Unit_class": 5, "PrototypeData_class": 5}',
                       help="Number of candidates for each class in JSON format")
    parser.add_argument("--rounds", type=json.loads, 
                       default='{"recognition": 1, "validation": 1}',
                       help="Number of rounds for each task in JSON format")

    parser.add_argument("--approach", type=str, default="base", choices=['base', 'proba'],
                       help="Approach to use for evaluation (default: 'base')")
    
    parser.add_argument("--validation_steps", 
        nargs='*', 
        default=[
            "prune_by_ontology",
            "supplement_empty_candidates",
            "candidate_probability",
            "select_class_by_prob",
        ]
    )
    
    # Tournament ranking parameters
    parser.add_argument("--pool_size", type=int, default=12,
                       help="Pool size for tournament ranking (default: 12)")
    parser.add_argument("--advance_ratio", type=float, default=1/6,
                       help="Advance ratio for tournament ranking (default: 0.1667)")
    # Caching options
    parser.add_argument("--save_recognition_file", type=str, 
                       help="Path to save recognition results for reuse")
    
    return parser


def configure_logging_for_evaluation() -> str:
    """
    Configure logging for evaluation and return the log file path.
    
    Returns:
        Path to the log file
    """
    log_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "..", 
        "..",
        "data_store", 
        "test_data", 
        "Annotated data", 
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    log_config.configure_logger(log_file_path=log_file_path)
    return log_file_path


def save_evaluation_results(
    evaluation_results: List[Dict[str, Any]], 
    metrics: Dict[str, Any], 
    recognition_save_path: str,
    args: argparse.Namespace,
    experiment_id: Optional[str] = None
) -> str:
    """
    Enhanced version with experiment tracking.
    
    Args:
        evaluation_results: List of evaluation results
        metrics: Calculated metrics
        recognition_save_path: Path to recognition cache file
        args: Command-line arguments
        experiment_id: Optional experiment identifier
        
    Returns:
        Path to the saved results file
    """
    # Generate unique experiment identifier
    if not experiment_id:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    result_json = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "approach": args.approach,
            "sample_size": args.sample_size,
            "threshold": args.threshold,
            "rounds": args.rounds,
            "model": args.model,
            "use_chain_of_thought": args.use_chain_of_thought,
            "number_of_candidates": args.number_of_candidates,
            "pool_size": args.pool_size,
            "advance_ratio": args.advance_ratio,
            "sample_non_none_only": args.sample_non_none_only,
            "sample_seed": args.sample_seed,
            "validation_steps": args.validation_steps
        },
        "recognition_cache_file": recognition_save_path,
        "evaluation_results": evaluation_results,
        "metrics": metrics,
        "summary": {
            "total_accuracy": metrics.get("total_accuracy", 0.0),
            "human_intervention_rate": metrics.get("human_intervention_rate", 0.0),
            "per_class_accuracies": {
                class_type: metrics.get("per_class_metrics", {}).get(class_type, {}).get("accuracy", 0.0)
                for class_type in ["PrototypeData", "Quantity", "Unit"]
            }
        }
    }

    save_file_prefix = os.path.basename(recognition_save_path).replace('.json', '') if recognition_save_path else "evaluation_results"
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "..",
        "..",
        "data_store",
        "test_data",
        "Annotated data",
        "evaluation_results",
        save_file_prefix
    )
    os.makedirs(result_dir, exist_ok=True)

    # Save results to JSON file
    results_file = os.path.join(
        result_dir,
        f"{save_file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    return results_file


def run_standard_evaluation(args: argparse.Namespace) -> tuple:
    """
    Run standard evaluation using run_single_task.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (evaluation_results, metrics, recognition_save_path)
    """
    logger.info("Starting standard system evaluation...")
    
    evaluation_results, metrics, recognition_save_path = run_evaluation(
        sample_size=args.sample_size,
        threshold=args.threshold,
        use_chain_of_thought=args.use_chain_of_thought,
        use_interpretation=args.dont_interpret,
        sample_non_none_only=args.sample_non_none_only,
        sample_seed=args.sample_seed,
        number_of_candidates=args.number_of_candidates,
        rounds=args.rounds,
        model=args.model,
        pool_size=args.pool_size,
        advance_ratio=args.advance_ratio,
        save_recognition_results_path=args.save_recognition_file,
        approach=args.approach,
        validation_steps=args.validation_steps
    )
    
    logger.info(f"Recognition results automatically saved to: {recognition_save_path}")
    
    return evaluation_results, metrics, recognition_save_path


def main():
    """
    Main entry point for the evaluation CLI.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging_for_evaluation()
 
    # Run standard evaluation (cached evaluation removed)
    evaluation_results, metrics, recognition_save_path = run_standard_evaluation(args)

    # Save results
    save_evaluation_results(evaluation_results, metrics, recognition_save_path, args)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
