"""
Single prediction evaluation utilities.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def evaluate_single_prediction(
    prediction: Dict[str, Any], 
    candidates: Dict[str, List[str]], 
    human_intervention: bool,
    label: Dict[str, Optional[str]],
    mnemonic: str
) -> Dict[str, Any]:
    """
    Evaluate a single prediction against its label.
    Updated to properly handle human_intervention flag from run_single_task.
    
    Args:
        prediction: Dictionary with predicted classes (Quantity_class, Unit_class, PrototypeData_class)
        candidates: Dictionary with candidate lists for each class
        human_intervention: Whether human intervention was needed (from run_single_task)
        label: Ground truth labels
        mnemonic: Mnemonic identifier for logging
        
    Returns:
        Dictionary with evaluation results including F1-ready data
    """
    class_types = ["PrototypeData", "Quantity", "Unit"]
    class_mapping = {
        "PrototypeData": "PrototypeData_class",
        "Quantity": "Quantity_class", 
        "Unit": "Unit_class"
    }
    
    results: Dict[str, Any] = {
        "mnemonic": mnemonic,
        "human_intervention_needed": human_intervention,
        "total_candidates": 0,
        "class_results": {},
        "f1_data": {}  # Structured data for F1 calculations
    }
    
    for class_type in class_types:
        pred_key = class_mapping[class_type]
        predicted_class = prediction.get(pred_key)
        true_label = label.get(class_type)
        candidate_key = pred_key.replace('_class', '_candidates')
        class_candidates = candidates.get(candidate_key, [])
        
        # Count candidates for reporting
        if human_intervention:
            results["total_candidates"] += len(class_candidates)
        
        # Evaluate correctness based on human_intervention flag
        if human_intervention:
            # When human intervention is needed, we evaluate based on candidates
            if true_label and true_label != "None":
                correct = true_label in class_candidates if class_candidates else False
                effective_prediction = true_label if correct else "None"
            else:
                correct = True  # No true label to find
                effective_prediction = "None"
        else:
            # Normal case: evaluate prediction directly
            if predicted_class is not None and predicted_class != "None":
                correct = predicted_class == true_label
                effective_prediction = predicted_class
            else:
                # No prediction made but no human intervention either
                correct = (true_label is None or true_label == "None")
                effective_prediction = "None"
        
        results["class_results"][class_type] = {
            "predicted": predicted_class,
            "true_label": true_label,
            "correct": correct,
            "candidates": class_candidates,
            "num_candidates": len(class_candidates)
        }
        
        # Store structured data for F1 calculations
        results["f1_data"][class_type] = {
            "true_label": true_label,
            "predicted_label": effective_prediction,
            "candidates": class_candidates,
            "has_prediction": predicted_class is not None and predicted_class != "None",
            "correct": correct
        }
    
    # Overall correctness: all classes must be correct
    results['correct'] = all(
        class_result['correct'] for class_result in results['class_results'].values()
    )
    
    return results
