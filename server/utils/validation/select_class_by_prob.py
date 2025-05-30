"""
select_class_by_prob.py

Validation utility for selecting class by probability with cutoff and threshold.
"""
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def select_class_by_prob(
    candidates: Dict[str, Dict[str, float]],
    recognized_class: Dict[str, str],
    cutoff: float = 0.0,
    threshold: float = 0.8,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    For each candidate class (Quantity, Unit, PrototypeData):
      - Remove candidates with probability below cutoff.
      - If the highest probability candidate is above threshold, set as recognized_class.
    Args:
        candidates: Dict of candidate lists with probabilities for each class type.
        recognized_class: Dict of current recognized classes.
        cutoff: Minimum probability to keep a candidate.
        threshold: Probability required to auto-select a class.
    Returns:
        (updated_candidates, updated_recognized_class)
    """
    updated_candidates = {}
    updated_recognized_class = recognized_class.copy()
    for class_key in ["Quantity_candidates", "Unit_candidates", "PrototypeData_candidates"]:
        class_candidates = candidates.get(class_key, {})
        # Remove candidates below cutoff
        filtered = {k: v for k, v in class_candidates.items() if v >= cutoff}
        updated_candidates[class_key] = filtered
        class_name = class_key.replace("_candidates", "_class")
        # Select class if highest prob >= threshold
        if filtered:
            best_candidate, best_prob = max(filtered.items(), key=lambda x: x[1])
            if best_prob >= threshold:
                logger.debug(f"Selected {class_name} '{best_candidate}' with probability {best_prob}")
                updated_recognized_class[class_name] = best_candidate
            else:
                logger.debug(f"Top candidate for {class_name} '{best_candidate}' below threshold {threshold}, setting to 'Uncertain'")
                updated_recognized_class[class_name] = "Uncertain" + class_name.replace("_class", "")
        else:
            updated_recognized_class[class_name] = "Uncertain" + class_name.replace("_class", "")
    logger.debug(f"Updated recognized classes: {updated_recognized_class}")
    return updated_candidates, updated_recognized_class

def select_top_candidates(candidates_with_prob: dict) -> dict:
    """
    Select the top candidate for each class type based on highest probability.
    
    Args:
        candidates_with_prob: Dict containing candidates with probabilities
        
    Returns:
        Dict with top candidate selected for each class type
    """
    selected_classes = {}
    
    # Select top candidates for each type
    for candidates_key in ["Quantity_candidates", "Unit_candidates", "PrototypeData_candidates"]:
        if candidates_key in candidates_with_prob:
            candidates = candidates_with_prob[candidates_key]
            if candidates and isinstance(candidates, dict):
                # Find candidate with highest probability
                top_candidate = max(candidates.items(), key=lambda x: x[1])
                class_key = candidates_key.replace("_candidates", "_class")
                selected_classes[class_key] = top_candidate[0]
            else:
                # No candidates available
                class_key = candidates_key.replace("_candidates", "_class")
                selected_classes[class_key] = "Uncertain" + class_key.replace("_class", "")
        else:
            # Key not found
            class_key = candidates_key.replace("_candidates", "_class")
            selected_classes[class_key] = "Uncertain" + class_key.replace("_class", "")

    logger.debug(f"Selected top candidates: {selected_classes}")
    return selected_classes