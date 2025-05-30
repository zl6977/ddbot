import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from . import chat_restapi as llm
from . import utils as ut
from .base_recognition import combined_runner as base_rec
from .configs import globals_config as glb
from .measure_utils import recognition_cache as rc
from .probability_recognition import combined_runner as proba_rec
from .probability_recognition import judge
from .validation import knowledge_distiller, validator  # noqa: F401

logger = logging.getLogger(__name__)

#############################
# Modular task registries
#############################

# Task function signatures
# - pre_recognition task: (task_key, task_data, models_high_low) -> dict (updated task_data)
# - post_recognition task: (task_key, task_data, recognition_output, models_high_low) -> tuple(recognized_class, candidates, additional_info)
# - user_interaction task: (project_id, task_data) -> dict (result info)

TaskFunc = Callable[..., Any]

TASK_REGISTRY: Dict[str, Dict[str, TaskFunc]] = {
    "pre_recognition": {},
    "post_recognition": {},
    "user_interaction": {},
}

DEFAULT_TASKS: Dict[str, List[str]] = {
    # Keep pre-step small by default; can be extended via TaskControl.pre_recognition_tasks
    "pre_recognition": ["interpret_mnemonic"],
    # For base approach, validator pipeline runs after recognition
    "post_recognition": ["validator_pipeline", "judge_assessment"],
    # After user confirm, distill knowledge and persist recognition for future retrieval
    "user_interaction": ["distill_knowledge", "save_recognition"],
}

def register_task(stage: str, name: str, func: TaskFunc) -> None:
    if stage not in TASK_REGISTRY:
        raise ValueError(f"Unknown stage '{stage}' for task registration")
    TASK_REGISTRY[stage][name] = func



def preprocess_free_user_input(free_user_input: str) -> dict:
    """
    preprocessed_metadata contains
        {
        "Mnemonic": "Mnemonic",
        "Description": "Description",
        "Unit": "unit_name",...}
    """
    preprocessed_metadata = {}
    prompt_template = glb.prompt_template_collection["Generic"]["Preprocessing"]["preprocessing"]
    kvPairs = {"<free_user_input>": free_user_input}
    prompt = ut.assemble_prompt(prompt_template, kvPairs)
    response = ut.run_rag_task_single(prompt, model=llm.DEFAULT_MODEL)
    preprocessed_metadata = response.get("content", {})
    return preprocessed_metadata


def initialize_task_batch(preprocessed_metadata_batch: dict) -> dict:
    """
    preprocessed_metadata contains
        {"Namespace": "project URI",
        "Mnemonic": "uid",
        "Unit": "unit_name",...}
    """

    task_template = {
        "Namespace": "",
        "Raw_content": "",
        "DrillingDataPoint_name": "",
        "PrototypeData_class": "",
        "PrototypeData_class_candidates": [],
        "MeasurableQuantity_class": "",
        "Quantity_class": "",
        "Quantity_class_candidates": "",
        "Unit_name": "",
        "Unit_class": "",
        "Unit_class_candidates": [],
        "Provider_name": "",
        "Interpretation_user": "",
        "TaskControl": {
            "Chain_of_Thought": False,
            "number_of_candidates": {
                "Quantity_class": 5,
                "Unit_class": 5,
                "PrototypeData_class": 5,
            },
            "approach" : "base",
            "pre_recognition_tasks": ["interpret_mnemonic"],
            "post_recognition_tasks": ["validator_pipeline"],
        },
        "ValidationControl": {
            "threshold": 0.5,
            "cutoff": 0.0,
            "validation_rounds": 2,
            "number_of_candidates": {
                "Quantity_class": 5,
                "Unit_class": 5,
                "PrototypeData_class": 5,
            },
            "max_iterations": 3,
            "validation_steps" : [
                "prune_by_ontology",
                "supplement_empty_candidates",
                "candidate_probability",
                "select_class_by_prob"
            ]
        },
    }
    task_batch = {}
    for pp_metadata in preprocessed_metadata_batch.values():
        task = task_template.copy()
        DrillingDataPoint_name = pp_metadata["Mnemonic"]
        task["Namespace"] = pp_metadata["Namespace"]
        task["DrillingDataPoint_name"] = DrillingDataPoint_name

        kick_out = ["Namespace"]
        raw_metadata = {k: v for k, v in pp_metadata.items() if k not in kick_out}
        task["Raw_content"] = json.dumps(raw_metadata)

        task_batch.update({DrillingDataPoint_name: task})
    return task_batch


#############################
# Task runners
#############################

def _get_task_list(task_data: dict, stage: str) -> List[str]:
    """Resolve configured tasks for a stage.

    Semantics:
    - If TaskControl has "<stage>_tasks":
        - If it's a list (including empty), return it. Empty list means run no tasks.
        - Otherwise, treat as invalid and run no tasks.
    - If the key is absent, fall back to DEFAULT_TASKS[stage].
    """
    ctrl = task_data.get("TaskControl", {})
    key = f"{stage}_tasks"
    if key in ctrl:
        val = ctrl.get(key)
        if isinstance(val, list):
            return val  # empty list => no tasks
        return []      # invalid type => no tasks
    return DEFAULT_TASKS.get(stage, [])


def run_pre_recognition_tasks(task_key: str, task_data: dict, models_high_low: List[str]) -> dict:
    """Execute configured pre-recognition tasks in order, returning updated task_data."""
    for name in _get_task_list(task_data, "pre_recognition"):
        func = TASK_REGISTRY["pre_recognition"].get(name)
        if not func:
            logger.warning(f"[pre] Task '{name}' not registered; skipping")
            continue
        try:
            task_data = func(task_key, task_data, models_high_low)
        except Exception as e:
            logger.exception(f"[pre] Task '{name}' failed: {e}")
    return task_data


def run_post_recognition_tasks(
    task_key: str,
    task_data: dict,
    recognition_output: tuple,
    models_high_low: List[str],
) -> tuple:
    """Execute configured post-recognition tasks in sequence, threading outputs.

    A post task may return a tuple(recognized_class, candidates, add_info) or a
    dict with keys to merge into add_info. Tuple outputs replace the current
    recognition_output for downstream tasks.
    """
    current_output: Optional[tuple] = recognition_output
    add_info_agg: Dict[str, Any] = {**task_data}
    for name in _get_task_list(task_data, "post_recognition"):
        func = TASK_REGISTRY["post_recognition"].get(name)
        if not func:
            logger.warning(f"[post] Task '{name}' not registered; skipping")
            continue
        try:
            out = func(task_key, task_data, current_output, models_high_low)
            if isinstance(out, tuple) and len(out) == 3:
                # Replace pipeline state
                current_output = out
            elif isinstance(out, dict):
                add_info_agg.update(out)
            elif out is None:
                pass
            else:
                logger.warning(f"[post] Task '{name}' returned unexpected type {type(out)}; ignoring")
        except Exception as e:
            logger.exception(f"[post] Task '{name}' failed: {e}")
    # Fallback if nothing produced
    if current_output is None:
        try:
            recognized_class, candidates, _prompts = recognition_output
        except Exception:
            recognized_class, candidates = {}, {}
        add_info_agg.setdefault("human_intervention_needed", True)
        return recognized_class, candidates, add_info_agg
    # Unpack and merge
    rcg, cand, _ = current_output
    return rcg, cand, add_info_agg


def run_user_interaction_tasks(project_id: str, task_data: dict) -> dict:
    """Execute configured user-interaction tasks; return final merged payload."""
    payload: Dict[str, Any] = {"project_id": project_id}
    for name in _get_task_list(task_data, "user_interaction"):
        func = TASK_REGISTRY["user_interaction"].get(name)
        if not func:
            logger.warning(f"[user] Task '{name}' not registered; skipping")
            continue
        try:
            out = func(project_id, task_data)
            if isinstance(out, dict):
                payload.update(out)
        except Exception as e:
            logger.exception(f"[user] Task '{name}' failed: {e}")
            payload.setdefault("errors", []).append({name: str(e)})
    return payload


#############################
# Concrete task implementations
#############################

def interpret_mnemonic_pre_task(task_key: str, task_data: dict, models_high_low: List[str]) -> dict:
    """Ensure Interpretation_user is populated via LLM if missing or empty."""
    interp, _ = interpret_mnemonic(
        task_data.get("Raw_content", ""),
        task_data.get("TaskControl", {}).get("Chain_of_Thought", False),
        model=models_high_low[0],
        approach=task_data.get("TaskControl", {}).get("approach", "base"),
    )
    task_data["Interpretation_user"] += interp
    return task_data


def validator_post_task(
    task_key: str,
    task_data: dict,
    recognition_output: tuple,
    models_high_low: List[str],
):
    """Run existing validator pipeline with recognized output."""
    user_query = task_data.get("Raw_content", "")
    interpretation = task_data.get("Interpretation_user", "")
    validation_control = task_data.get("ValidationControl", {})
    threshold = validation_control.get("threshold", 0.5)
    cutoff = validation_control.get("cutoff", 0.0)
    validation_round = validation_control.get("validation_rounds", 1)
    validation_tasks = validation_control.get("validation_steps", [])

    validation_results = validator.validate_metadata_with_defaults(
        recognition_output=recognition_output,
        metadata=user_query,
        interpretation=interpretation,
        model=models_high_low[0],
        rounds=validation_round,
        validation_tasks=validation_tasks,
        cutoff=cutoff,
        threshold=threshold,
    )

    human_intervention_needed = validator.signal_human_intervention_needed(
        recognized_class=validation_results["recognized_class"],
        considered_classes=["Quantity_class", "Unit_class", "PrototypeData_class"],
    )

    return (
        validation_results["recognized_class"],
        validation_results["candidates"],
        {"human_intervention_needed": human_intervention_needed, "approach": task_data.get("TaskControl", {}).get("approach", "base")},
    )


def judge_post_task(
    task_key: str,
    task_data: dict,
    recognition_output: tuple,
    models_high_low: List[str],
):
    """Run judging on recognition output to assess quality and human need.

    Returns a dict merged into add_info; does not alter recognition outputs.
    """
    try:
        recognized_class, candidates, _prompts = recognition_output
    except Exception:
        logger.warning("[judge] Invalid recognition_output; skipping judge")
        return {"judge_error": "invalid_recognition_output"}

    user_query = task_data.get("Raw_content", "")
    interpretation = task_data.get("Interpretation_user", "")
    validation_control = task_data.get("ValidationControl", {})
    confidence_threshold = validation_control.get("threshold", 0.5)
    validation_rounds = validation_control.get("validation_rounds", 1)

    enriched = ut.enrich_classes_with_extra_content(recognized_class)
    try:
        judge_score, judge_info = judge.judge_recognition_quality(
            mnemonic=user_query,
            recognized_class=enriched,
            candidates=candidates,
            interpretation=interpretation,
            model=models_high_low[0],
            rounds=validation_rounds,
        )
        human_needed, reason = judge.assess_human_intervention_needed(
            recognized_class=recognized_class,
            candidates=candidates,
            confidence_score=judge_score,
            threshold=confidence_threshold,
            additional_criteria=judge_info,
        )
        return {
            "human_intervention_needed": human_needed,
            "judge_score": judge_score,
            "judge_reason": reason,
        }
    except Exception as e:
        logger.exception(f"[judge] judge failed: {e}")
        return {"judge_error": str(e)}


def distill_knowledge_user_task(project_id: str, task_data: dict) -> dict:
    user_query = task_data.get("Raw_content", "")
    interpretation = task_data.get("Interpretation_user", "")
    recognized_class = {
        "Quantity_class": task_data.get("Quantity_class", "None"),
        "Unit_class": task_data.get("Unit_class", "None"),
        "PrototypeData_class": task_data.get("PrototypeData_class", "None"),
    }
    try:
        distill_knowledge(
            user_query=user_query,
            interpretation=interpretation,
            recognized_class=recognized_class,
            model=llm.DEFAULT_MODEL,
        )
        return {"knowledge_distilled": True}
    except Exception as e:
        logger.exception("distill_knowledge failed in user task")
        return {"knowledge_distilled": False, "error_distill": str(e)}


def save_recognition_user_task(project_id: str, task_data: dict) -> dict:
    """Persist the user's confirmed recognition for future retrieval."""

    # Build a minimal cache with single sample
    dp_name = task_data.get("DrillingDataPoint_name") or task_data.get("Raw_content", "")[:50]
    recognition_cache = {
        dp_name: {
            "recognized_class": {
                "Quantity_class": task_data.get("Quantity_class", "None"),
                "Unit_class": task_data.get("Unit_class", "None"),
                "PrototypeData_class": task_data.get("PrototypeData_class", "None"),
            },
            "raw": task_data.get("Raw_content", ""),
            "interpretation": task_data.get("Interpretation_user", ""),
        }
    }

    params = {
        "model": llm.DEFAULT_MODEL,
        "use_chain_of_thought": task_data.get("TaskControl", {}).get("Chain_of_Thought", False),
        "number_of_candidates": task_data.get("TaskControl", {}).get("number_of_candidates", {}),
        "recognition_rounds": task_data.get("TaskControl", {}).get("recognition_rounds", 1),
    }

    # Save into project task folder if possible
    tasks_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")
    project_dir = os.path.join(tasks_root, project_id) if project_id else tasks_root
    os.makedirs(project_dir, exist_ok=True)
    filepath = os.path.join(project_dir, f"recognition_{dp_name}.json")
    ok = rc.save_recognition_results(recognition_cache, filepath, params)
    return {"recognition_saved": ok, "recognition_path": filepath}


# Register concrete tasks
register_task("pre_recognition", "interpret_mnemonic", interpret_mnemonic_pre_task)
register_task("post_recognition", "validator_pipeline", validator_post_task)
register_task("post_recognition", "judge_assessment", judge_post_task)
register_task("user_interaction", "distill_knowledge", distill_knowledge_user_task)
register_task("user_interaction", "save_recognition", save_recognition_user_task)

def run_single_task(
    task_key: str,
    task_data: dict,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
    approach: str | None = None,
) -> tuple:
    """Unified dispatcher selecting between baseline and probabilistic implementations.

        Precedence to determine approach:
            1. Explicit 'approach' argument
            2. task_data['TaskControl']['approach']
            3. task_data['approach']
            4. Default: 'base'
    """
    if approach is None:
        approach = (
            task_data.get("TaskControl", {}).get("approach")
            or task_data.get("approach")
            or "base"
        )
    approach = approach.lower()

    # Run pre-recognition tasks centrally
    task_data = run_pre_recognition_tasks(task_key, task_data, models_high_low)

    if approach in {"proba", "prob", "iter", "iterative"}:
        recognition_output = proba_rec.recognize_metadata(task_key, task_data, models_high_low)
    else:
        recognition_output = base_rec.recognize_metadata(task_key, task_data, models_high_low)


    recognized_class, candidates, add_info = run_post_recognition_tasks(
            task_key, task_data, recognition_output, models_high_low
        )
    
    return recognized_class, candidates, add_info


    # obsolete recognize_metadata helper removed (moved into proba combined_runner)

def interpret_mnemonic(
    user_query: str,
    CoT_flag: bool,
    model: str = llm.DEFAULT_MODEL,
    approach: str = "base"
) -> tuple:
    if approach in {"proba", "prob", "iter", "iterative"}:
        interpretation, prompt = proba_rec.interpret_mnemonic(user_query, CoT_flag, model)
    else:
        interpretation, prompt = base_rec.interpret_mnemonic(user_query, CoT_flag, model)

    # interpretation, prompt = narrow_selection_range("Interpret_mnemonic", prompt_templates, metadata)
    return interpretation, prompt

def distill_knowledge(
    user_query: str,
    interpretation: str,
    recognized_class: dict,
    model: str
):
    # Implement the knowledge distillation logic here
    # This is a placeholder implementation
    has_new_insight, knowledge_entry = knowledge_distiller.distill_knowledge(
        user_query=user_query,
        interpretation=interpretation,
        recognized_class=recognized_class,
        model=model
    )
    if has_new_insight and knowledge_entry:
        if knowledge_distiller.save_knowledge_to_complementary_knowledge(knowledge_entry):
            knowledge_distiller.update_complementary_knowledge_in_memory(knowledge_entry)


def handle_user_interaction(project_id: str, task_data: dict) -> dict:
    """Entry point to handle user confirmation actions after recognition.

    Runs the configured user_interaction tasks and returns a payload for the API layer.
    """
    payload = run_user_interaction_tasks(project_id, task_data)
    # Always include a friendly message and default flags expected by UI
    payload.setdefault("message", "User interaction tasks executed.")
    payload.setdefault("human_intervention_needed", False)
    return payload
