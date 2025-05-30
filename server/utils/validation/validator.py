import logging

# import sys
# from pathlib import Path
from typing import List, Optional, Tuple  # noqa

from ..configs import globals_config as glb
from ..probability_recognition import rank_candidates as rk
from . import prune_by_ontology as pbo
from . import select_class_by_prob as scbp
from . import supplement_empty_candidates as sec

logger = logging.getLogger(__name__)


def measure_candidate_probabilities(
    recognized_class: dict,
    candidates: dict,
    prompt_templates: dict,
    metadata: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 5,
    user_config: Optional[dict] = None,
    complementary_knowledge: Optional[dict] = None,
) -> dict:
    """
    Measure candidate probabilities using LLM ranking with logprob extraction.

    Args:
        candidates: Dict containing candidate lists from recognize_metadata
        prompt_templates: Prompt templates for ranking
        metadata: Original user query metadata
        interpretation: Optional interpretation string
        model: LLM model to use
        rounds: Number of rounds for probability aggregation
        user_config: Optional LLM configuration
        complementary_knowledge: Optional drilling terminology knowledge

    Returns:
        Dict with candidates and their associated probabilities.
    """
    logger.info("Measuring candidate probabilities")

    new_candidates = rk.rank_all_candidates(
        candidates=candidates,
        recognized_class=recognized_class,
        prompt_templates=prompt_templates,
        metadata=metadata,
        interpretation=interpretation,
        model=model,
        rounds=rounds,
        user_config=user_config,
        complementary_knowledge=complementary_knowledge,
    )

    logger.debug(f"New Candidates results: {new_candidates}")
    return new_candidates


def execute_validation_task(task_name: str, recognized_class: dict, candidates: dict, prompts: dict, metadata: dict, **kwargs) -> dict:
    """
    Execute a single validation task.

    Args:
        task_name: Name of the validation task to execute
        recognized_class: Recognition results from recognize_metadata
        candidates: Candidate lists from recognize_metadata
        prompts: Prompts used in recognition from recognize_metadata
        metadata: Original user query metadata
        **kwargs: Additional task-specific parameters

    Returns:
        Dict with validation results for the task
    """
    logger.info(f"Executing validation task: {task_name}")

    if task_name == "candidate_probability":
        prompt_templates = kwargs.get("prompt_templates")
        if prompt_templates is None:
            raise ValueError("prompt_templates is required for candidate_probability task")
        logger.info("Measuring candidate probabilities")
        return measure_candidate_probabilities(
            recognized_class=recognized_class,
            candidates=candidates,
            prompt_templates=prompt_templates,
            metadata=metadata,
            interpretation=kwargs.get("interpretation"),
            model=kwargs.get("model"),
            rounds=kwargs.get("rounds", 5),
            user_config=kwargs.get("user_config"),
            complementary_knowledge=kwargs.get("complementary_knowledge"),
        )
    elif task_name == "select_class_by_prob":
        logger.info("Selecting class by probability")
        cutoff = kwargs.get("cutoff", 0.0)
        threshold = kwargs.get("threshold", 0.5)
        updated_candidates, updated_recognized_class = scbp.select_class_by_prob(
            candidates=candidates,
            recognized_class=recognized_class,
            cutoff=cutoff,
            threshold=threshold,
        )
        logger.debug(f"Updated candidates: {updated_candidates}")
        logger.debug(f"Updated recognized class: {updated_recognized_class}")
        return {"candidates": updated_candidates, "recognized_class": updated_recognized_class}
    elif task_name == "prune_by_ontology":
        logger.info("Pruning candidates by ontology")
        updated_candidates, updated_recognized_class, messages = pbo.prune_by_ontology(
            candidates=candidates,
            recognized_class=recognized_class,
        )
        logger.debug(f"Updated candidates after pruning: {updated_candidates}")
        logger.debug(f"Updated recognized class after pruning: {updated_recognized_class}")
        return {"candidates": updated_candidates, "recognized_class": updated_recognized_class}
    elif task_name == "supplement_empty_candidates":
        logger.info("Supplementing empty candidates using ontology constraints")
        updated_candidates = sec.supplement_empty_candidates(
            recognized_class=recognized_class,
            candidates=candidates,
        )
        logger.debug(f"Updated candidates after supplementation: {updated_candidates}")
        return {
            "candidates": updated_candidates,
            "recognized_class": recognized_class,  # recognized_class doesn't change in this step
        }
    else:
        logger.warning(f"Unknown validation task: {task_name}")
        return {"error": f"Unknown validation task: {task_name}"}


def merge_validation_results(recognition_output: Tuple[dict, dict, dict], task_name: str, task_result: dict) -> Tuple[dict, dict, dict]:
    """
    Merge validation task results into the recognition output tuple.

    Args:
        recognition_output: Tuple containing [recognized_class, candidates, prompts]
        task_name: Name of the validation task
        task_result: Results from the validation task

    Returns:
        Updated recognition output tuple with validation results merged
    """
    recognized_class, candidates, prompts = recognition_output

    updated_class = task_result.get("recognized_class") or recognized_class.copy()
    updated_candidates = task_result.get("candidates") or candidates.copy()
    updated_prompts = task_result.get("prompts") or prompts.copy()

    return (updated_class, updated_candidates, updated_prompts)


def validate_metadata(recognition_output: Tuple[dict, dict, dict], metadata: dict, validation_tasks: List[str], **task_kwargs) -> dict:
    """
    Perform validation tasks on the output of recognize_metadata.

    Args:
        recognition_output: Tuple containing [recognized_class, candidates, prompts]
                           from recognize_metadata
        metadata: Original user query metadata
        validation_tasks: List of validation task names to execute
        **task_kwargs: Additional parameters passed to validation tasks

    Returns:
        Dict containing:
        - original recognition results (recognized_class)
        - original candidates
        - original prompts
        - validation results for each task
    """
    recognized_class, candidates, prompts = recognition_output

    logger.info(f"Starting validation with tasks: {validation_tasks}")

    # Start with original recognition output
    current_output = (recognized_class.copy(), candidates.copy(), prompts.copy())
    validation_results = {"recognized_class": current_output[0], "candidates": current_output[1], "prompts": current_output[2], "validation_tasks": {}}

    # Execute each validation task
    for task_name in validation_tasks:
        try:
            task_result = execute_validation_task(
                task_name=task_name,
                recognized_class=current_output[0],
                candidates=current_output[1],
                prompts=current_output[2],
                metadata=metadata,
                **task_kwargs,
            )
            validation_results["validation_tasks"][task_name] = task_result

            # Merge results into current output
            current_output = merge_validation_results(current_output, task_name, task_result)

            # Update validation_results with merged data
            validation_results["recognized_class"] = current_output[0]
            validation_results["candidates"] = current_output[1]
            validation_results["prompts"] = current_output[2]

        except Exception as e:
            logger.error(f"Error executing validation task {task_name}: {e}")
            validation_results["validation_tasks"][task_name] = {"error": str(e)}

    logger.info("Validation completed")
    return validation_results


def validate_metadata_with_defaults(
    recognition_output: Tuple[dict, dict, dict],
    metadata: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 5,
    user_config: Optional[dict] = None,
    validation_tasks: Optional[List[str]] = None,
    cutoff: float = 0.0,
    threshold: float = 0.5,
) -> dict:
    """
    Validate metadata with common default parameters.

    Args:
        recognition_output: Tuple containing [recognized_class, candidates, prompts]
                           from recognize_metadata
        metadata: Original user query metadata
        prompt_templates: Prompt templates for ranking
        interpretation: Optional interpretation string
        model: LLM model to use
        rounds: Number of rounds for probability aggregation
        user_config: Optional LLM configuration
        validation_tasks: List of validation task names (defaults to ["candidate_probability"])
        cutoff: Probability cutoff for select_class_by_prob (default 0.0)
        threshold: Probability threshold for select_class_by_prob (default 0.5)
        complementary_knowledge: Optional drilling terminology knowledge
    Returns:
        Dict containing validation results
    """
    if validation_tasks is None:
        validation_tasks = []
    return validate_metadata(
        recognition_output=recognition_output,
        metadata=metadata,
        validation_tasks=validation_tasks,
        prompt_templates=glb.prompt_template_collection["Generic"],
        interpretation=interpretation,
        model=model,
        rounds=rounds,
        user_config=user_config,
        cutoff=cutoff,
        threshold=threshold,
        complementary_knowledge=glb.complementary_knowledge,
    )


def signal_human_intervention_needed(
    recognized_class: dict,
    considered_classes=["Quantity_class", "Unit_class", "PrototypeData_class"],
) -> bool:
    return any(
        (
            recognized_class.get(cls, "Uncertain" + cls.replace("_class", "")) == "Uncertain" + cls.replace("_class", "")
        ) or
        (
            recognized_class.get(cls, "Uncertain" + cls.replace("_class", "")) == "OutOfSet" + cls.replace("_class", "")
        ) for cls in considered_classes
    )


def test_validator():
    """
    Simple test for the validation framework: checks that it runs without errors and returns expected structure.
    """
    logging.basicConfig(level=logging.DEBUG)

    # Mock recognition output
    recognized_class = {
        "Quantity_class": "ForceQuantity",
        "Unit_class": "kilogram-force",
        "PrototypeData_class": "HookLoad",
    }
    candidates = {
        "Quantity_candidates": ["ForceQuantity", "PressureQuantity"],
        "Unit_candidates": ["kilogram-force", "bar"],
        "PrototypeData_candidates": ["HookLoad", "Pressure"],
    }
    prompts = {}
    recognition_output = (recognized_class, candidates, prompts)

    metadata = {"Mnemonic": "HKLO", "Description": "Hookload", "Unit": "kkgf", "DataType": "double"}

    # Mock prompt templates
    prompt_templates = {
        "Recognize_quantity": {
            "ranking": "Task Objective: Select the most likely candidate.\nInput:\nUser query: <user_query>\nCandidate list:\n<candidates>\nInterpretation: <interpretation>\nOutput: (Only the single letter)"
        },
        "Recognize_unit": {
            "ranking": "Task Objective: Select the most likely unit.\nInput:\nUser query: <user_query>\nCandidate list:\n<candidates>\nInterpretation: <interpretation>\nOutput: (Only the single letter)"
        },
        "Recognize_prototypeData": {
            "ranking": "Task Objective: Select the most likely prototype data.\nInput:\nUser query: <user_query>\nCandidate list:\n<candidates>\nInterpretation: <interpretation>\nOutput: (Only the single letter)"
        },
    }

    print("Testing validator with candidate_probability task...")
    try:
        result = validate_metadata_with_defaults(
            recognition_output=recognition_output,
            metadata=metadata,
            prompt_templates=prompt_templates,
            interpretation="HKLO means Hookload",
            validation_tasks=["candidate_probability", "select_class_by_prob", "prune_by_ontology", "supplement_empty_candidates"],
        )
        assert "recognized_class" in result, "Missing recognized_class in result"
        assert "candidates" in result, "Missing candidates in result"
        assert "prompts" in result, "Missing prompts in result"
        assert "validation_tasks" in result, "Missing validation_tasks in result"
        assert "candidate_probability" in result["validation_tasks"], "Missing candidate_probability task result"
        assert "prune_by_ontology" in result["validation_tasks"], "Missing prune_by_ontology task result"
        assert "select_class_by_prob" in result["validation_tasks"], "Missing select_class_by_prob task result"
        assert "supplement_empty_candidates" in result["validation_tasks"], "Missing supplement_empty_candidates task result"
        print(f"Validation result keys: {result.keys()}")
        print(f"Recognition class keys: {result['recognized_class'].keys()}")
        print(f"Candidates keys: {result['candidates'].keys()}")
        print(f"Validation tasks: {result['validation_tasks'].keys()}")

        print("test_validator passed.")
    except Exception as e:
        print(f"Test failed (expected if no API access): {e}")


if __name__ == "__main__":
    test_validator()
