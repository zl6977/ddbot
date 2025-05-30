"""
Standard evaluation execution utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .. import task_manager
from .data_loader import (
    convert_input_to_task_format,
    load_test_data,
    prepare_labeled_samples,
)
from .metrics_calculator import calculate_metrics
from .prediction_evaluator import evaluate_single_prediction
from .recognition_cache import get_cache_file_path, save_recognition_results

logger = logging.getLogger(__name__)


def run_evaluation(
    sample_size: Optional[int] = None,
    threshold: float = 0.5,
    use_chain_of_thought: bool = False,
    use_interpretation: bool = False,
    sample_non_none_only: bool = False,
    sample_seed: int = 42,
    number_of_candidates: Dict[str, int] = {"Quantity_class": 5, "Unit_class": 10, "PrototypeData_class": 5},
    rounds: Dict[str, int] = {"recognition": 2, "validation": 2},
    model: str = 'gpt-4o-mini',
    save_recognition_results_path: Optional[str] = None,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
    approach: str = 'base',
    validation_steps: List[str] = [
        "candidate_probability",
        "select_class_by_prob",
        "prune_by_ontology",
        "supplement_empty_candidates",
    ]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Run evaluation on test data using run_single_task and save recognition results.
    
    Args:
        sample_size: Number of samples to evaluate (None for all)
        threshold: Probability threshold for validation
        use_chain_of_thought: Whether to use Chain of Thought reasoning
        use_interpretation: Whether to use interpret_mnemonic for better interpretation
        sample_non_none_only: Whether to sample only mnemonics whose labels are not None
        sample_seed: Random seed for sampling
        number_of_candidates: Number of candidates for each class in JSON format
        rounds: Number of rounds for each task in JSON format
        distill_knowledge: Whether to distill knowledge
        model: LLM model to use
        save_recognition_results_path: Optional path to save recognition results
        
    Returns:
        Tuple of (evaluation_results, metrics, recognition_save_path)
    """
    # Store recognition parameters for compatibility checking
    recognition_params = {
        "sample_size": sample_size,
        "use_chain_of_thought": use_chain_of_thought,
        "use_interpretation": use_interpretation,
        "sample_non_none_only": sample_non_none_only,
        "sample_seed": sample_seed,
        "number_of_candidates": number_of_candidates,
        "recognition_rounds": rounds.get("recognition", 2),
        "model": model,
        "pool_size": pool_size,
        "advance_ratio": advance_ratio
    }
    
    # Load test data
    input_data, label_data = load_test_data()
    
    # Prepare labeled samples
    labeled_samples = prepare_labeled_samples(
        input_data, label_data, sample_non_none_only, sample_size, sample_seed
    )
    
    logger.info(f"Evaluating {len(labeled_samples)} samples with threshold={threshold}")
    
    # Initialize task manager
    # task_manager.load_files()
    
    evaluation_results = []
    recognition_cache = {}
    
    for i, sample in enumerate(labeled_samples):
        mnemonic = sample["mnemonic"]
        logger.debug(f"Processing sample {i+1}/{len(labeled_samples)}: {mnemonic}")
        
        try:
            # Convert to task format with parameters
            task_data = convert_input_to_task_format(
                sample,
                approach=approach,
                use_chain_of_thought=use_chain_of_thought,
                use_interpretation=use_interpretation,
                number_of_candidates=number_of_candidates,
                recognition_rounds=rounds.get("recognition", 2),
                pool_size=pool_size,
                advance_ratio=advance_ratio,
                # validation Args
                validation_rounds=rounds.get("validation", 2),
                threshold=threshold,
                validation_steps=validation_steps
            )
            
            # Run prediction
            recognized_class, candidates, meta_info = task_manager.run_single_task(mnemonic, task_data, [model, model], approach)
            human_intervention = meta_info.get("human_intervention_needed", False)

            # Store recognition result for caching
            recognition_cache[mnemonic] = {
                "recognition_result": {
                    "recognized_class": recognized_class,
                    "candidates": candidates,
                    "meta_info": meta_info,
                    "task_data": task_data
                },
                "label": label_data[mnemonic],
                "sample": sample
            }
            
            # Get ground truth
            label = label_data[mnemonic]
            
            # Evaluate prediction
            result = evaluate_single_prediction(
                recognized_class, candidates, human_intervention, label, mnemonic
            )
            evaluation_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {mnemonic}: {e}")
            continue
    
    # Calculate metrics
    metrics = calculate_metrics(evaluation_results)

    logger.info(f"Evaluation completed. Total Accuracy: {metrics['total_accuracy']:.3f}")
    
    # Safely access metrics with defaults
    micro_metrics = metrics.get('micro_metrics', {})
    macro_metrics = metrics.get('macro_metrics', {})
    per_class_metrics = metrics.get('per_class_metrics', {})
    
    logger.info(f"Micro F1: {micro_metrics.get('f1', 0.0):.3f}, Macro F1: {macro_metrics.get('f1', 0.0):.3f}")
    logger.info(f"Human Intervention Rate: {metrics['human_intervention_rate']:.3f}, "
                f"Avg Candidates on Intervention: {metrics['avg_candidates_on_intervention']:.1f}")
    
    # Log per-class performance
    for class_type in ["PrototypeData", "Quantity", "Unit"]:
        class_metrics = per_class_metrics.get(class_type, {})
        logger.info(f"{class_type} - Accuracy: {class_metrics.get('accuracy', 0.0):.3f}, "
                   f"F1-micro: {class_metrics.get('f1_micro', 0.0):.3f}, "
                   f"F1-macro: {class_metrics.get('f1_macro', 0.0):.3f}")
    
    # Always save recognition results (either to provided path or auto-generated)
    if save_recognition_results_path:
        actual_save_path = save_recognition_results_path
    else:
        actual_save_path = get_cache_file_path()
    
    save_recognition_results(recognition_cache, actual_save_path, recognition_params)
    logger.info(f"Recognition results saved to: {actual_save_path}")
    
    return evaluation_results, metrics, actual_save_path
