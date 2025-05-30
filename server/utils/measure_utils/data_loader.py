"""
Data loading and preprocessing utilities for system evaluation.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


def load_test_data() -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load input data and label data from test files.
    
    Returns:
        Tuple containing (input_data_list, label_data_dict)
    """
    utils_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(utils_dir, "..", "..", "data_store", "test_data", "Annotated data")
    
    # Load input data
    input_file = os.path.join(data_dir, "input_data.yaml")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = yaml.safe_load(f)
    
    # Load label data
    label_file = os.path.join(data_dir, "label_data.yaml")
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = yaml.safe_load(f)
    
    logger.info(f"Loaded {len(input_data)} input samples and {len(label_data)} label entries")
    return input_data, label_data


def convert_input_to_task_format(
    input_sample: Dict[str, str],
    approach: str = "base",
    use_interpretation: bool = True,
    use_chain_of_thought: bool = False,
    number_of_candidates: Dict[str, int] = {"Quantity_class": 5, "Unit_class": 10, "PrototypeData_class": 5},
    recognition_rounds: int = 2,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
    threshold: float = 0.5, 
    validation_rounds: int = 2,
    validation_steps: List[str] = ["candidate_probability", "select_class_by_prob", "prune_by_ontology", "supplement_empty_candidates"]
) -> Dict[str, Any]:
    """
    Convert input data sample to the format expected by run_single_task.
    
    Args:
        input_sample: Dictionary with mnemonic, description, unit
        threshold: Probability threshold for validation
        cutoff: Cutoff value for validation
        use_chain_of_thought: Whether to use Chain of Thought reasoning
        interpretation: User interpretation string
        number_of_candidates: Number of candidates for each class (recognition parameter)
        recognition_rounds: Number of rounds for recognition
        validation_rounds: Number of rounds for validation
        validation_steps: List of validation steps to perform
        distill_knowledge: Whether to distill knowledge
        pool_size: Maximum candidates per pool for tournament ranking
        advance_ratio: Fraction of candidates advancing in tournament
        
    Returns:
        Task data dictionary compatible with run_single_task
    """
    # Create preprocessed metadata format
    preprocessed_metadata = {
        "Namespace": "http://test.example.org/",
        "Mnemonic": input_sample["mnemonic"],
        "Description": input_sample.get("description", ""),
        "Unit": input_sample.get("unit", ""),
    }
    
    # Convert to task format
    raw_content = json.dumps({k: v for k, v in preprocessed_metadata.items() if k != "Namespace"})
    
    task_data = {
        "Namespace": preprocessed_metadata["Namespace"],
        "Raw_content": raw_content,
        "DrillingDataPoint_name": input_sample["mnemonic"],
        "Interpretation_user": "",
        "TaskControl": {
            "approach": approach,
            "pre_recognition_tasks" : ["interpret_mnemonic"] if use_interpretation else [],
            "post_recognition_tasks": ["validator_pipeline"] if approach in {"base", "baseline"} else ["judge_assessment"],
            "Chain_of_Thought": use_chain_of_thought,
            "number_of_candidates": number_of_candidates,
            "recognition_rounds": recognition_rounds,
            "pool_size": pool_size,
            "advance_ratio": advance_ratio,
        },
        "ValidationControl": {
            "threshold": threshold,
            "validation_rounds": validation_rounds,
            "number_of_candidates": number_of_candidates,
            "validation_steps": validation_steps
        }
    }
    
    return task_data


def prepare_labeled_samples(
    input_data: List[Dict],
    label_data: Dict[str, Dict],
    sample_non_none_only: bool = False,
    sample_size: Optional[int] = None,
    sample_seed: int = 42
) -> List[Dict]:
    """
    Filter and sample input data based on label availability and requirements.
    
    Args:
        input_data: List of input samples
        label_data: Dictionary mapping mnemonics to labels
        sample_non_none_only: Whether to exclude samples with None labels
        sample_size: Number of samples to return (None for all)
        sample_seed: Random seed for sampling
        
    Returns:
        List of filtered and sampled input data
    """
    
    # Filter input data to only include samples that have labels
    labeled_samples = []
    for sample in input_data:
        mnemonic = sample["mnemonic"]
        if mnemonic in label_data:
            mnemonic_label = label_data[mnemonic]
            if any(value == "None" for value in mnemonic_label.values()) and sample_non_none_only:
                # Skip samples where any label is None
                continue
            labeled_samples.append(sample)
    
    if sample_size:
        random.seed(sample_seed)
        labeled_samples = random.sample(labeled_samples, min(sample_size, len(labeled_samples)))
    
    return labeled_samples
