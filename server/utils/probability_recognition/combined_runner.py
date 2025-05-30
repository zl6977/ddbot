import logging
from typing import Any, Dict, List, Tuple

from .. import chat_restapi as llm
from ..configs import globals_config as glb
from ..validation import prune_by_ontology, select_class_by_prob  # type: ignore
from . import recognizer_pc as rec_pc
from . import recognizer_pc_cot as rec_pc_cot

logger = logging.getLogger(__name__)


def interpret_mnemonic(
    user_query: str,
    CoT_flag: bool,
    model: str = llm.DEFAULT_MODEL,
):
    if CoT_flag:
        prompt_templates = glb.prompt_template_collection["PC_CoT"]
        interpretation, prompt = rec_pc_cot.interpret_mnemonic(
            user_query, glb.complementary_knowledge, prompt_templates, model
        )
    else:
        prompt_templates = glb.prompt_template_collection["PC"]
        interpretation, prompt = rec_pc.interpret_mnemonic(
            user_query, glb.complementary_knowledge, prompt_templates, model
        )
    return interpretation, prompt


def recognize_metadata(
    task_key: str,
    task_data: dict,
    models_high_low: List[str] = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Iterative probabilistic recognition without judging.

    Returns a triple (recognized_class, candidates_with_prob, meta) for post-processing.
    """
    logger.debug(f"[proba] Running recognition for task {task_key} (iterative loop, judge deferred)")

    # Extract task parameters
    user_query = task_data.get("Raw_content", "")
    interpretation = task_data.get("Interpretation_user", "")
    validation_control = task_data.get("ValidationControl", {})
    task_control = task_data.get("TaskControl", {})

    # Loop control / thresholds
    max_iterations = validation_control.get("max_iterations", 5)

    # Recognition parameters
    CoT_flag = task_control.get("Chain_of_Thought", False)
    number_of_candidates = validation_control.get(
        "number_of_candidates",
        task_control.get(
            "number_of_candidates",
            {"Quantity_class": 5, "Unit_class": 10, "PrototypeData_class": 5},
        ),
    )
    recognition_rounds = task_control.get("recognition_rounds", 1)
    pool_size = task_control.get("pool_size", 12)
    advance_ratio = task_control.get("advance_ratio", 1 / 6)

    extra_messages: List[str] = []
    iteration_count = 0
    # candidates_with_prob structure: {
    #   "Quantity_candidates": {"ForceQuantity": 0.8, ...},
    #   "Unit_candidates": {"kilogram-force": 0.7, ...},
    #   "PrototypeData_candidates": {"HookLoad": 0.6, ...}
    # }
    candidates_with_prob: Dict[str, Dict[str, float]] = {}
    updated_classes: Dict[str, Any] = {}

    while iteration_count < max_iterations:
        iteration_count += 1
        logger.debug(f"[proba] Iteration {iteration_count}/{max_iterations} for task {task_key}")

        current_interpretation = interpretation
        if extra_messages:
            messages_str = "\n\nAdditional context from previous iterations:\n" + "\n".join(extra_messages)
            current_interpretation = (interpretation or "") + messages_str + "\n"

        # Run probabilistic recognition using PC or PC_CoT recognizers
        if CoT_flag:
            prompt_templates = glb.prompt_template_collection["PC_CoT"]
            recognition_output = rec_pc_cot.recoginize_metadata(
                user_query,  # type: ignore
                current_interpretation,
                glb.complementary_knowledge,
                prompt_templates,
                glb.quantity_fullList_extraContent,
                glb.unit_fullList_extraContent,
                glb.prototypeData_fullList_extraContent,
                models_high_low,
                number_of_candidates,
                rounds=recognition_rounds,
                pool_size=pool_size,
                advance_ratio=advance_ratio,
            )
        else:
            prompt_templates = glb.prompt_template_collection["PC"]
            recognition_output = rec_pc.recoginize_metadata(
                user_query,  # type: ignore
                current_interpretation,
                glb.complementary_knowledge,
                prompt_templates,
                glb.quantity_fullList_extraContent,
                glb.unit_fullList_extraContent,
                glb.prototypeData_fullList_extraContent,
                models_high_low,
                number_of_candidates,
                rounds=recognition_rounds,
                use_extra_content=False,
                pool_size=pool_size,
                advance_ratio=advance_ratio,
            )

        # Extract candidates from recognition output tuple
        _recognized_class_raw, candidates, _prompts = recognition_output
        candidates_with_prob = candidates

        # Select and prune
        recognized_class = select_class_by_prob.select_top_candidates(candidates_with_prob)
        _, updated_classes, pruning_messages = prune_by_ontology.prune_by_ontology(
            candidates=candidates_with_prob,
            recognized_class=recognized_class,
        )

        if recognized_class != updated_classes:
            logger.debug(f"[proba] Validation failed in iteration {iteration_count}")
            extra_messages.extend(pruning_messages)
            continue

        meta = {
            "extra_messages": extra_messages,
            "iterations_used": iteration_count,
            "validation_approach": "iterative_loop",
            "approach": "proba",
        }
        return updated_classes, candidates_with_prob, meta

    logger.warning(f"[proba] Task {task_key} reached maximum iterations ({max_iterations}) without success")
    meta = {
        "extra_messages": extra_messages,
        "iterations_used": max_iterations,
        "validation_approach": "iterative_loop",
        "approach": "proba",
    }
    return updated_classes, candidates_with_prob, meta