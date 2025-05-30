"""
Supplement empty candidates using ontology constraints.

This module provides functions to fill in missing candidates for semantic classes
by using ontological relationships when other classes are already determined.
"""

import logging
from typing import Dict, List, Optional, Set

from .. import sparql_connector as sc
from ..configs import globals_config as glb

logger = logging.getLogger(__name__)


def identify_empty_candidate_types(candidates: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Identify which candidate types have only "None" as candidates.
    
    Args:
        candidates: Dict of candidate lists with probabilities for each class type
        
    Returns:
        List of candidate type names that need supplementing
    """
    empty_types = []
    
    candidate_mappings = {
        "Quantity_candidates": "Quantity",
        "PrototypeData_candidates": "PrototypeData", 
        "Unit_candidates": "Unit"
    }
    
    for candidates_key, class_type in candidate_mappings.items():
        class_candidates = candidates.get(candidates_key, {})
        
        # Check if only "None" exists or if all non-None candidates have zero probability
        non_none_candidates = {k: v for k, v in class_candidates.items() if k not in ["OutOfSet" + class_type, "Uncertain" + class_type]}

        if not non_none_candidates or all(prob == 0.0 for prob in non_none_candidates.values()):
            empty_types.append(class_type)
            logger.debug(f"Found empty candidate type: {class_type}")
    
    return empty_types


def get_selected_class_value(recognized_class: Dict[str, str], class_type: str) -> str:
    """
    Get the selected class value for a given type.
    
    Args:
        recognized_class: Dict with currently recognized classes
        class_type: Type of class to get value for
        
    Returns:
        Selected class value or "None" if not set
    """
    class_mappings = {
        "Quantity": "Quantity_class",
        "PrototypeData": "PrototypeData_class", 
        "Unit": "Unit_class"
    }
    
    class_key = class_mappings.get(class_type)
    if class_key:
        return recognized_class.get(class_key, "Uncertain" + class_type)
    return "Uncertain" + class_type


def get_quantity_candidates_from_constraints(
    unit_class: str, 
    prototypedata_class: str,
) -> List[str]:
    """
    Get valid Quantity candidates based on Unit and PrototypeData constraints.
    
    Args:
        unit_class: Selected Unit class ("None" if not set)
        prototypedata_class: Selected PrototypeData class ("None" if not set)
        
    Returns:
        List of valid Quantity candidates
    """
    # Get candidates from each constraint
    candidates_from_unit = set()
    candidates_from_prototype = set()
    
    candidates_from_unit = set(glb.unit_fullList_extraContent.get(unit_class, {}).get("ddhub:IsUnitForQuantity", []))
    logger.debug(f"Quantities from unit {unit_class}: {candidates_from_unit}")

    candidates_from_prototype = set(glb.prototypeData_fullList_extraContent.get(prototypedata_class, {}).get("ddhub:IsOfBaseQuantity", []))
    logger.debug(f"Quantities from prototypedata {prototypedata_class}: {candidates_from_prototype}")
    
    # Determine final candidates based on constraints
    if candidates_from_unit and candidates_from_prototype:
        # Both constraints exist - use intersection
        final_candidates = candidates_from_unit.intersection(candidates_from_prototype)
        logger.debug(f"Intersection of quantity candidates: {final_candidates}")
    elif candidates_from_unit:
        # Only unit constraint
        final_candidates = candidates_from_unit
    elif candidates_from_prototype:
        # Only prototype constraint
        final_candidates = candidates_from_prototype
    else:
        # No constraints
        final_candidates = set()
    
    return list(final_candidates)


def get_unit_candidates_from_constraints(
    quantity_class: str,
    prototypedata_class: str
) -> List[str]:
    """
    Get valid Unit candidates based on Quantity and PrototypeData constraints.
    
    Args:
        quantity_class: Selected Quantity class ("None" if not set)
        prototypedata_class: Selected PrototypeData class ("None" if not set)
        
    Returns:
        List of valid Unit candidates
    """
    candidates_from_quantity = set()

    candidates_from_quantity = set(glb.quantity_fullList_extraContent.get(quantity_class, {}).get("zzz:QuantityHasUnit", []))
    logger.debug(f"Units from quantity {quantity_class}: {candidates_from_quantity}")
    
    # Note: PrototypeData doesn't directly constrain Units in the current ontology
    # Units are constrained through Quantity relationships
    
    return list(candidates_from_quantity)


def get_prototypedata_candidates_from_constraints(
    quantity_class: str,
    unit_class: str
) -> List[str]:
    """
    Get valid PrototypeData candidates based on Quantity and Unit constraints.
    
    Args:
        quantity_class: Selected Quantity class ("None" if not set)
        unit_class: Selected Unit class ("None" if not set)
        
    Returns:
        List of valid PrototypeData candidates
    """
    candidates_from_quantity = set()
    
    candidates_from_quantity = set(glb.quantity_fullList_extraContent.get(quantity_class, {}).get("zzz:PrototypeData", []))
    logger.debug(f"PrototypeData from quantity {quantity_class}: {candidates_from_quantity}")
    
    # Note: Unit doesn't directly constrain PrototypeData in the current ontology
    # PrototypeData is constrained through Quantity relationships
    
    return list(candidates_from_quantity)


def supplement_empty_candidates(
    recognized_class: Dict[str, str],
    candidates: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Supplement empty candidate lists using ontology constraints.
    
    Takes recognized classes and candidate probabilities, and for any candidate type
    that has only "None" values, fills in compatible candidates based on ontological
    relationships with other determined classes.
    
    Example:
        If Unit="kilogram-force" and PrototypeData="HookLoad" are selected,
        but Quantity_candidates only contains "None", this function will:
        1. Query valid quantities for the unit "kilogram-force" 
        2. Query valid quantities for the prototype data "HookLoad"
        3. Take the intersection (since both constraints exist)
        4. Add compatible quantities like "ForceQuantity" with equal probabilities
    
    Args:
        recognized_class: Dict with currently recognized classes 
            Format: {"Quantity_class": "ForceQuantity", "Unit_class": "None", ...}
        candidates: Dict of candidate lists with probabilities for each class type
            Format: {"Quantity_candidates": {"None": 1.0}, "Unit_candidates": {...}, ...}
        
    Returns:
        Updated candidates dict with supplemented empty types
        
    Note:
        - Only supplements types that have only "None" or zero-probability candidates
        - Uses intersection of constraints when multiple classes are selected
        - Distributes equal probability among new candidates (including "None")
        - Follows functional programming style - input dicts are not modified
    """
    
    logger.info("Starting candidate supplementation")
    
    # Create a copy to avoid modifying the original
    updated_candidates = candidates.copy()
    
    # Find which types need supplementing
    empty_types = identify_empty_candidate_types(candidates)
    
    if not empty_types:
        logger.debug("No empty candidate types found")
        return updated_candidates
    
    logger.info(f"Found empty candidate types: {empty_types}")
    
    # Get current class selections
    quantity_class = get_selected_class_value(recognized_class, "Quantity")
    unit_class = get_selected_class_value(recognized_class, "Unit")
    prototypedata_class = get_selected_class_value(recognized_class, "PrototypeData")
    
    logger.debug(f"Current selections - Quantity: {quantity_class}, Unit: {unit_class}, PrototypeData: {prototypedata_class}")
    
    # Supplement each empty type
    for empty_type in empty_types:
        new_candidates = []
        candidates_key = ""  # Initialize to avoid unbound variable
        
        if empty_type == "Quantity":
            new_candidates = get_quantity_candidates_from_constraints(
                unit_class, prototypedata_class
            )
            candidates_key = "Quantity_candidates"
            
        elif empty_type == "Unit":
            new_candidates = get_unit_candidates_from_constraints(
                quantity_class, prototypedata_class
            )
            candidates_key = "Unit_candidates"
            
        elif empty_type == "PrototypeData":
            new_candidates = get_prototypedata_candidates_from_constraints(
                quantity_class, unit_class
            )
            candidates_key = "PrototypeData_candidates"
        
        # Add new candidates with equal probability distribution
        if new_candidates and candidates_key:
            logger.info(f"Adding {len(new_candidates)} candidates for {empty_type}: {new_candidates}")
            
            # Calculate equal probability for each new candidate
            total_candidates = len(new_candidates)
            equal_prob = 1.0 / total_candidates
            
            # Create new candidate dict
            new_candidate_dict = { }
            for candidate in new_candidates:
                new_candidate_dict[candidate] = equal_prob
                
            updated_candidates[candidates_key] = new_candidate_dict
            
        else:
            logger.info(f"No compatible candidates found for {empty_type}")
    
    logger.info("Candidate supplementation completed")
    return updated_candidates
