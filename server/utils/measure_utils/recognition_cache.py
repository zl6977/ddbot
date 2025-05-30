"""
Recognition result caching and persistence utilities.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def save_recognition_results(
    recognition_cache: Dict[str, Any], 
    filepath: str,
    recognition_params: Dict[str, Any]
) -> bool:
    """
    Save recognition results to disk with metadata.
    
    Args:
        recognition_cache: Recognition results cache from run_evaluation
        filepath: Path to save the recognition results file
        recognition_params: Parameters used for recognition (model, CoT, rounds, etc.)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        save_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(recognition_cache),
                "recognition_params": recognition_params,
                "format_version": "1.0"
            },
            "recognition_cache": recognition_cache
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Recognition results saved to {filepath} ({len(recognition_cache)} samples)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save recognition results to {filepath}: {e}")
        return False


def load_recognition_results(filepath: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load recognition results from disk.
    
    Args:
        filepath: Path to the recognition results file
        
    Returns:
        Tuple of (recognition_cache, recognition_params) or (None, None) if failed
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Recognition results file not found: {filepath}")
            return None, None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        if "recognition_cache" not in save_data or "metadata" not in save_data:
            logger.error(f"Invalid recognition results file format: {filepath}")
            return None, None
            
        recognition_cache = save_data["recognition_cache"]
        metadata = save_data["metadata"]
        recognition_params = metadata.get("recognition_params", {})

        logger.info(f"Loaded recognition results from {filepath} ({len(recognition_cache)} samples)")
        return recognition_cache, recognition_params
        
    except Exception as e:
        logger.error(f"Failed to load recognition results from {filepath}: {e}")
        return None, None


def validate_recognition_compatibility(
    recognition_params: Dict[str, Any],
    current_params: Dict[str, Any]
) -> bool:
    """
    Validate that stored recognition results are compatible with current evaluation parameters.
    
    Args:
        recognition_params: Parameters used when recognition results were generated
        current_params: Current evaluation parameters
        
    Returns:
        True if compatible, False otherwise
    """
    critical_params = ['model', 'use_chain_of_thought', 'number_of_candidates', 'recognition_rounds']
    
    for param in critical_params:
        if recognition_params.get(param) != current_params.get(param):
            logger.warning(f"Critical parameter mismatch: {param} "
                         f"(stored: {recognition_params.get(param)}, "
                         f"current: {current_params.get(param)})")
            return False
    
    return True


def get_cache_file_path() -> str:
    """
    Generate a cache file path based on parameters and timestamp.
    
    Args:
        base_name: Base name for the cache file
        params: Optional parameters to include in the filename hash
        
    Returns:
        Generated file path with timestamp and parameter hash
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}.json"
    
    utils_dir = os.path.dirname(os.path.dirname(__file__))
    cache_dir = os.path.join(utils_dir, "..", "..", "data_store", "test_data", "recognition_cache")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, filename)
