"""
Comprehensive metrics calculation utilities for system evaluation.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics including micro/macro F1, precision, recall for each class type.
    Handles the special case where None predictions are evaluated against candidates.
    
    Args:
        evaluation_results: List of evaluation results from evaluate_single_prediction
        
    Returns:
        Dictionary with comprehensive metrics including per-class and aggregated metrics
    """
    total_samples = len(evaluation_results)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "per_class_metrics": {},
            "micro_metrics": {},
            "macro_metrics": {},
            "mnemonic_metrics": {},
            "human_intervention_metrics": {}
        }
    
    class_types = ["PrototypeData", "Quantity", "Unit"]
    
    # Collect all unique labels across all classes for proper metric calculation
    all_true_labels: Dict[str, List[str]] = {}
    all_pred_labels: Dict[str, List[str]] = {}
    all_labels_set: Dict[str, set] = {}
    
    # Initialize label collections
    for class_type in class_types:
        all_true_labels[class_type] = []
        all_pred_labels[class_type] = []
        all_labels_set[class_type] = set()
    
    # Human intervention tracking
    human_interventions = 0
    total_candidates_during_intervention = 0
    mnemonic_correct = 0
    
    # Process each evaluation result
    for result in evaluation_results:
        # Mnemonic-level accuracy (all classes correct)
        if result.get("correct", False):
            mnemonic_correct += 1
        
        # Human intervention tracking
        if result["human_intervention_needed"]:
            human_interventions += 1
            total_candidates_during_intervention += result.get("total_candidates", 0)
        
        # Process each class for F1 calculations
        for class_type in class_types:
            f1_data = result["f1_data"][class_type]
            true_label = f1_data["true_label"]
            predicted_label = f1_data["predicted_label"]
            candidates = f1_data["candidates"]
            has_prediction = f1_data["has_prediction"]
            
            # Handle None values consistently
            if true_label is None or true_label == "None":
                true_label = "None"
            if predicted_label is None or predicted_label == "None":
                predicted_label = "None"
            
            # Key update: For human intervention cases, evaluate based on candidates
            # If human intervention is needed, we check if the true label is in candidates
            if result["human_intervention_needed"] and candidates and true_label != "None":
                # For human intervention: if true label is in candidates, count as "found"
                if true_label in candidates:
                    predicted_label = true_label  # Treat as correct prediction for metrics
                else:
                    predicted_label = "None"  # Not found in candidates
            elif not has_prediction and candidates and true_label != "None":
                # Legacy case: when no prediction but we have candidates
                if true_label in candidates:
                    predicted_label = true_label
                else:
                    predicted_label = "None"
            
            all_true_labels[class_type].append(true_label)
            all_pred_labels[class_type].append(predicted_label)
            all_labels_set[class_type].add(true_label)
            all_labels_set[class_type].add(predicted_label)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    
    for class_type in class_types:
        y_true = all_true_labels[class_type]
        y_pred = all_pred_labels[class_type]
        labels = sorted(list(all_labels_set[class_type]))
        
        # Calculate basic accuracy (original logic)
        correct_predictions = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        accuracy = correct_predictions / len(y_true) if len(y_true) > 0 else 0.0
        
        # Calculate precision, recall, F1 using sklearn
        try:
            precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
            recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
            
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
            
            class_report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
        except Exception as e:
            logger.warning(f"Error calculating metrics for {class_type}: {e}")
            precision_micro = precision_macro = recall_micro = recall_macro = f1_micro = f1_macro = 0.0
            class_report = {}
        
        per_class_metrics[class_type] = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": len(y_true),
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "classification_report": class_report,
            "unique_labels": labels
        }
    
    # Calculate overall micro metrics (pool all classes together)
    all_y_true = []
    all_y_pred = []
    for class_type in class_types:
        all_y_true.extend(all_true_labels[class_type])
        all_y_pred.extend(all_pred_labels[class_type])
    
    overall_labels = sorted(list(set(all_y_true + all_y_pred)))
    
    try:
        overall_precision_micro = precision_score(all_y_true, all_y_pred, average='micro', zero_division=0, labels=overall_labels)
        overall_recall_micro = recall_score(all_y_true, all_y_pred, average='micro', zero_division=0, labels=overall_labels)
        overall_f1_micro = f1_score(all_y_true, all_y_pred, average='micro', zero_division=0, labels=overall_labels)
        
        overall_accuracy = sum(1 for i in range(len(all_y_true)) if all_y_true[i] == all_y_pred[i]) / len(all_y_true)
        
    except Exception as e:
        logger.warning(f"Error calculating overall metrics: {e}")
        overall_precision_micro = 0.0
        overall_recall_micro = 0.0
        overall_f1_micro = 0.0
        overall_accuracy = 0.0
    
    # Calculate macro metrics by averaging per-class metrics
    avg_precision_macro = np.mean([per_class_metrics[ct]["precision_macro"] for ct in class_types])
    avg_recall_macro = np.mean([per_class_metrics[ct]["recall_macro"] for ct in class_types])
    avg_f1_macro = np.mean([per_class_metrics[ct]["f1_macro"] for ct in class_types])
    avg_accuracy = np.mean([per_class_metrics[ct]["accuracy"] for ct in class_types])
    
    # Human intervention metrics
    human_intervention_rate = human_interventions / total_samples
    avg_candidates_on_intervention = (
        total_candidates_during_intervention / human_interventions 
        if human_interventions > 0 else 0.0
    )
    
    # Mnemonic-level metrics
    mnemonic_accuracy = mnemonic_correct / total_samples
    
    return {
        # Per-class detailed metrics
        "per_class_metrics": per_class_metrics,
        
        # Micro-averaged metrics (pooled across all classes)
        "micro_metrics": {
            "precision": overall_precision_micro,
            "recall": overall_recall_micro,
            "f1": overall_f1_micro,
            "accuracy": overall_accuracy
        },
        
        # Macro-averaged metrics (average of per-class metrics)
        "macro_metrics": {
            "precision": avg_precision_macro,
            "recall": avg_recall_macro,
            "f1": avg_f1_macro,
            "accuracy": avg_accuracy
        },
        
        # Mnemonic-level metrics (based on all classes being correct)
        "mnemonic_metrics": {
            "accuracy": mnemonic_accuracy,
            "correct_mnemonics": mnemonic_correct
        },
        
        # Human intervention metrics
        "human_intervention_metrics": {
            "rate": human_intervention_rate,
            "avg_candidates_on_intervention": avg_candidates_on_intervention,
            "total_interventions": human_interventions
        },
        
        # Summary statistics
        "total_samples": total_samples,
        "total_predictions": len(all_y_true),
        "total_correct_predictions": sum(1 for i in range(len(all_y_true)) if all_y_true[i] == all_y_pred[i])
    }


def calculate_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall accuracy, human intervention rate, and average candidates on intervention.
    Also calculates per-class metrics and mnemonic-level metrics.
    
    Args:
        evaluation_results: List of evaluation results from evaluate_single_prediction
        
    Returns:
        Dictionary with calculated metrics including per-class and mnemonic-level metrics
    """
    # Use the new comprehensive metrics function
    comprehensive_metrics = calculate_comprehensive_metrics(evaluation_results)
    
    # Return in the original format for backward compatibility, but with enhanced data
    result = {
        # Per-class metrics (enhanced)
        "per_class_metrics": comprehensive_metrics["per_class_metrics"],
        
        # Mnemonic-level metrics
        "mnemonic_metrics": comprehensive_metrics["mnemonic_metrics"],
        
        # Human intervention metrics (original format)
        "human_intervention_rate": comprehensive_metrics["human_intervention_metrics"]["rate"],
        "avg_candidates_on_intervention": comprehensive_metrics["human_intervention_metrics"]["avg_candidates_on_intervention"],
        
        # Summary statistics (original format)
        "total_samples": comprehensive_metrics["total_samples"],
        "total_predictions": comprehensive_metrics["total_predictions"],
        "total_correct": comprehensive_metrics["total_correct_predictions"],
        "total_accuracy": comprehensive_metrics["micro_metrics"]["accuracy"],
        "human_interventions": comprehensive_metrics["human_intervention_metrics"]["total_interventions"],
        
        # New: Add comprehensive metrics for detailed analysis
        "micro_metrics": comprehensive_metrics["micro_metrics"],
        "macro_metrics": comprehensive_metrics["macro_metrics"],
        "comprehensive": comprehensive_metrics
    }
    
    return result
