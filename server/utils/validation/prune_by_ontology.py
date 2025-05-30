"""
prune_by_ontology.py

Validation utility for pruning candidates based on ontology constraints.
"""
import logging
from typing import Callable, Dict, List, Tuple

from ..configs import globals_config as glb

logger = logging.getLogger(__name__)

def prune_quantity_candidates_against_prototypedata(quantity_candidates: dict, quantity_class: str, prototype_class: str) -> Tuple[dict, str]:
    """Prune Quantity candidates based on PrototypeData constraints."""
    message = ""

    if prototype_class in ["OutOfSetPrototypeData", "UncertainPrototypeData"]  or not quantity_candidates:
        return quantity_candidates.copy(), message
    
    valid_quantities = glb.prototypeData_fullList_extraContent.get(prototype_class, {}).get("ddhub:IsOfBaseQuantity", [])
    logger.debug(f"Valid quantities for PrototypeData '{prototype_class}': {valid_quantities}")
    if quantity_class not in valid_quantities:
        message = f"Quantity class '{quantity_class}' is incompatible with PrototypeData '{prototype_class}'"

    return {
        k: v for k, v in quantity_candidates.items() 
        if k in valid_quantities or k in ["OutOfSetQuantity", "UncertainQuantity"]
    }, message


def prune_quantity_candidates_against_unit(quantity_candidates: dict, quantity_class: str, unit_class: str) -> Tuple[dict, str]:
    """Prune Quantity candidates based on Unit constraints."""
    message = ""
    if unit_class in ["OutOfSetUnit", "UncertainUnit"] or not quantity_candidates:
        return quantity_candidates.copy(), message

    valid_quantities = glb.unit_fullList_extraContent.get(unit_class, {}).get("ddhub:IsUnitForQuantity", [])
    logger.debug(f"Valid quantities for unit '{unit_class}': {valid_quantities}")
    if quantity_class not in valid_quantities:
        message = f"Quantity class '{quantity_class}' is incompatible with Unit '{unit_class}'"

    return {
        k: v for k, v in quantity_candidates.items() 
        if k in valid_quantities or k in ["OutOfSetQuantity", "UncertainQuantity"]
    }, message
    


def prune_prototypedata_candidates_against_quantity(prototypedata_candidates: dict, prototypedata_class: str, quantity_class: str) -> Tuple[dict, str]:
    """Prune PrototypeData candidates based on Quantity constraints."""
    message = ""
    if quantity_class in ["OutOfSetQuantity", "UncertainQuantity"] or not prototypedata_candidates:
        return prototypedata_candidates.copy(), message

    valid_prototypedata = glb.quantity_fullList_extraContent.get(quantity_class, {}).get("zzz:PrototypeData", [])
    logger.debug(f"Valid PrototypeData for quantity '{quantity_class}': {valid_prototypedata}")
    if prototypedata_class not in valid_prototypedata:
        message = f"PrototypeData class '{prototypedata_class}' is incompatible with Quantity '{quantity_class}'"

    return {
        k: v for k, v in prototypedata_candidates.items() 
        if k in valid_prototypedata or k in ["OutOfSetPrototypeData", "UncertainPrototypeData"]
    }, message

def prune_unit_candidates_against_quantity(unit_candidates: dict, unit_class: str, quantity_class: str) -> Tuple[dict, str]:
    """Prune Unit candidates based on Quantity constraints."""
    message = ""
    if quantity_class in ["OutOfSetQuantity", "UncertainQuantity"] or not unit_candidates:
        return unit_candidates.copy(), message

    valid_units = glb.quantity_fullList_extraContent.get(quantity_class, {}).get("zzz:QuantityHasUnit", [])
    logger.debug(f"Valid units for quantity '{quantity_class}': {valid_units}")
    if unit_class not in valid_units:
        message = f"Unit class '{unit_class}' is incompatible with Quantity '{quantity_class}'"

    return {
        k: v for k, v in unit_candidates.items() 
        if k in valid_units or k in ["OutOfSetUnit", "UncertainUnit"]
    }, message


def get_pruning_operations():
    """
    Define all possible pruning operations with their dependencies.
    
    Returns:
        List of tuples: (target_candidates_key, pruning_function, constraint_class_key, description)
    """
    return [
        ("Quantity_candidates", "Quantity_class", prune_quantity_candidates_against_prototypedata, "PrototypeData_class", "Quantity against PrototypeData"),
        ("Quantity_candidates", "Quantity_class", prune_quantity_candidates_against_unit, "Unit_class", "Quantity against Unit"),
        ("PrototypeData_candidates", "PrototypeData_class", prune_prototypedata_candidates_against_quantity, "Quantity_class", "PrototypeData against Quantity"),
        ("Unit_candidates", "Unit_class", prune_unit_candidates_against_quantity, "Quantity_class", "Unit against Quantity"),
    ]


def get_pruning_order_by_probability(candidates: Dict[str, Dict[str, float]], recognized_class: Dict[str, str]) -> list:
    """
    Determine pruning order based on class probabilities.
    
    Args:
        candidates: Dict of candidate lists with probabilities for each class type.
        recognized_class: Dict of current recognized classes.
        
    Returns:
        List of class types ordered by probability (highest first)
    """
    class_mappings = {
        "Quantity_class": "Quantity_candidates",
        "PrototypeData_class": "PrototypeData_candidates", 
        "Unit_class": "Unit_candidates"
    }
    
    class_probs = []
    for class_key, candidates_key in class_mappings.items():
        class_value = recognized_class.get(class_key, "Uncertain" + class_key.replace("_class", ""))
        prob = candidates.get(candidates_key, {}).get(class_value, 0.0)
        class_probs.append((prob, class_key))
    
    # Sort by probability (highest first), then return just the class types
    class_probs.sort(key=lambda x: x[0], reverse=True)
    return [class_type for prob, class_type in class_probs]


def execute_pruning_step(
        updated_candidates: dict,
        updated_recognized_class: dict, 
        target_candidate: str,
        target_class: str,
        pruning_func: Callable,
        constraint_class_key: str,
        description: str
    ) -> str:
    """
    Execute a single pruning step and validate the result.
    
    Args:
        updated_candidates: Current candidates dict
        updated_recognized_class: Current recognized classes dict
        target_key: Key for candidates to be pruned (e.g., "Quantity_candidates")
        pruning_func: Function to perform the pruning
        constraint_class_key: Key for the constraining class (e.g., "PrototypeData_class")
        description: Description for logging
        
    Returns:
        Updated recognized_class dict after validation
    """
    constraint_value = updated_recognized_class.get(constraint_class_key, "Uncertain" + constraint_class_key.replace("_class", ""))

    # Perform pruning
    updated_candidates[target_candidate], message = pruning_func(
        updated_candidates.get(target_candidate, {}),
        updated_recognized_class.get(target_class, "Uncertain" + target_class.replace("_class", "")),
        constraint_value
    )
    logger.debug(f"Pruned {description} '{constraint_value}': {updated_candidates[target_candidate]}")

    if message:
        updated_recognized_class[target_class] = "Uncertain" + target_class.replace("_class", "")

    return message


def _compute_top_valid_combinations(
    candidates: Dict[str, Dict[str, float]],
    top_n: int = 3,
) -> List[Tuple[str, str, str, float]]:
    """Compute top-N valid (Quantity, Unit, PrototypeData) combinations by combined probability.

    A combination is valid if the Quantity is valid for both the Unit and the PrototypeData.
    Combined probability is naive product P(Q)*P(U)*P(PD) (independence assumption) to give a
    ranking signal; values are renormalized only for ranking purposes (not required here).

    Args:
        candidates: dict with keys 'Quantity_candidates', 'Unit_candidates', 'PrototypeData_candidates'.
        top_n: number of combinations to return.

    Returns:
        List of tuples (Quantity, Unit, PrototypeData, combined_probability) sorted desc by prob.
    """
    qty_cands = candidates.get("Quantity_candidates", {}) or {}
    unit_cands = candidates.get("Unit_candidates", {}) or {}
    proto_cands = candidates.get("PrototypeData_candidates", {}) or {}

    if not (qty_cands and unit_cands and proto_cands):
        return []

    def valid_quantities_for_unit(u: str) -> set:
        return set(glb.unit_fullList_extraContent.get(u, {}).get("ddhub:IsUnitForQuantity", ["OutOfSetQuantity", "UncertainQuantity"]))

    def valid_quantities_for_prototype(p: str) -> set:
        return set(glb.prototypeData_fullList_extraContent.get(p, {}).get("ddhub:IsOfBaseQuantity", ["OutOfSetQuantity", "UncertainQuantity"]))

    combinations: List[Tuple[str, str, str, float]] = []
    for q, qp in qty_cands.items():
        for u, up in unit_cands.items():
            # Quick fail: quantity must belong to unit's valid set
            if q not in valid_quantities_for_unit(u):
                continue
            for p, pp in proto_cands.items():
                if q not in valid_quantities_for_prototype(p):
                    continue
                combined = qp * up * pp
                combinations.append((q, u, p, combined))

    # Sort by combined probability descending
    combinations.sort(key=lambda x: x[3], reverse=True)
    return combinations[:top_n]


def prune_by_ontology(
    candidates: Dict[str, Dict[str, float]],
    recognized_class: Dict[str, str],
    suggestions_top_n: int = 3,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], List[str]]:
    """
    Prune candidates based on ontology constraints.
    Remove candidates that are not valid according to the ontology relationships.
    Pruning order is determined by the highest probability recognized class first.
    
    Args:
        candidates: Dict of candidate lists with probabilities for each class type.
        recognized_class: Dict of current recognized classes.
        suggestions_top_n: Number of top suggestions to return.

    Returns:
        (updated_candidates, updated_recognized_class, all_pruning_messages)
    """
    # Initialize with copies to avoid modifying original data
    updated_candidates = {k: v.copy() for k, v in candidates.items()}
    updated_recognized_class = recognized_class.copy()

    all_pruning_messages: List[str] = []
    
    # Get pruning order based on probabilities
    class_order = get_pruning_order_by_probability(candidates, recognized_class)
    logger.debug(f"Pruning order by probability: {class_order}")
    
    # Get all possible pruning operations
    all_operations = get_pruning_operations()
    
    # If we have a high-probability class, use it as anchor for initial pruning
    for i, class_type in enumerate(class_order):
        logger.debug(f"{i+1}. Pruning: Using '{class_type}' as anchor for pruning")
        
        # First, prune other classes against the anchor
        for target_candidate, target_class, pruning_func, constraint_class_key, description in all_operations:
            if constraint_class_key == class_type:
                message = execute_pruning_step(
                    updated_candidates, 
                    updated_recognized_class, 
                    target_candidate, 
                    target_class, 
                    pruning_func, 
                    constraint_class_key, 
                    description
                )
                if message:
                    all_pruning_messages.append(message)



    if all_pruning_messages:
        all_pruning_messages.insert(0, f"\nValidation failed: {recognized_class} are not valid under ontology constraints.")
        # Provide suggestion of valid combinations with highest probability
        suggestions = _compute_top_valid_combinations(candidates, top_n=suggestions_top_n)
        if suggestions:
            suggestion_lines = [
                "Suggested valid combinations (top {} by combined probability):".format(len(suggestions))
            ]
            for idx, (q, u, p, prob) in enumerate(suggestions, start=1):
                suggestion_lines.append(
                    f"  {idx}. Quantity={q} | Unit={u} | PrototypeData={p} (combined_prob={prob:.4f})"
                )
            all_pruning_messages.extend(suggestion_lines)
        else:
            all_pruning_messages.append(
                "No valid (Quantity, Unit, PrototypeData) combination found among current candidates; regenerate or broaden candidates."
            )
        all_pruning_messages.append(
            "Please change parts of the recognition to obtain valid recognition. This could be done by selecting different candidates or adjusting the recognized classes."
        )

    return updated_candidates, updated_recognized_class, all_pruning_messages
