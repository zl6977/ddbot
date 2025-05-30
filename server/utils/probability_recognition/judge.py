"""LLM-based judging utilities (relocated)."""
import json
import logging
from typing import Dict, List, Optional, Tuple

from .. import chat_restapi as llm
from .. import utils as ut
from ..configs import globals_config as glb
from . import rank_candidates as rank_utils

logger = logging.getLogger(__name__)


def judge_recognition_quality(
    mnemonic: str,
    recognized_class: dict,
    candidates: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 1,
    user_config: Optional[dict] = None,
) -> Tuple[float, dict]:
    if model is None:
        model = llm.DEFAULT_MODEL
    additional_info = {
        "has_none_values": any(v == "None" for v in recognized_class.values()),
        "total_candidates": sum(len(v) for v in candidates.values() if isinstance(v, dict)),
    }
    try:
        score_map = generic_llm_judge(
            mnemonic=mnemonic,
            recognized_class=recognized_class,
            interpretation=interpretation,
            model=model,
            rounds=rounds,
            user_config=user_config,
            options=["Yes", "No"],
        )
        final_score = score_map.get("Yes", 0.0)
    except Exception as e:  # pragma: no cover
        logger.warning(f"LLM-based judgment failed: {e}; returning 0.0")
        final_score = 0.0
        score_map = {"Yes": 0.0, "No": 1.0}
    additional_info.update({"distribution": score_map, "final_score": final_score})
    return final_score, additional_info


def generic_llm_judge(
    mnemonic: str,
    recognized_class: dict,
    interpretation: Optional[str],
    model: str,
    rounds: int,
    user_config: Optional[dict],
    options: Optional[List[str]] = None,
) -> Dict[str, float]:
    if options is None:
        options = ["Yes", "No"]
    prob_agg = {c: [] for c in options}

    judge_prompt_set = glb.prompt_template_collection['Generic']["QualityJudge"]
    system_prompt = judge_prompt_set.get("system", "")

    kv_analysis = judge_prompt_set.get("placeholders", [])
    kv_analysis = {k: "" for k in kv_analysis}

    stages = judge_prompt_set["stages"]
    tmpl_analysis = stages["analysis"]
    tmpl_extraction = stages["extraction"]


    complementary_knowledge_str = glb.complementary_knowledge.get("basic", "")
    recognized_class_str = json.dumps(recognized_class, ensure_ascii=False, indent=2)
    for _ in range(rounds):
        letter_map, reverse_map = ut.create_letter_mapping(options, model)
        options_str = "\n".join(f"{letter}: {cand}" for cand, letter in letter_map.items())
        kv_analysis.update({
            "<user_query>": mnemonic,
            "<interpretation>": interpretation or "",
            "<recognized_class>": recognized_class_str,
            "<complementary_knowledge>": complementary_knowledge_str,
        })
        analysis_prompt = ut.assemble_prompt(tmpl_analysis, kv_analysis)
        analysis_result = ut.run_rag_task_single(analysis_prompt, system_prompt=system_prompt, model=model, user_config={"temperature": 1.0})
        last_answer = analysis_result.get("content", "") if analysis_result else ""
        kv_extract = dict(kv_analysis)
        kv_extract.update({"<last_answer>": last_answer, "<options>": options_str})
        extraction_prompt = ut.assemble_prompt(tmpl_extraction, kv_extract)
        config = rank_utils.prepare_llm_config(letter_map, model, user_config)
        result = ut.run_rag_task_single(extraction_prompt, system_prompt=system_prompt, model=model, user_config=config)
        toplog_probs = rank_utils.extract_logprobs(result)
        rank_utils.update_prob_aggregation(toplog_probs, reverse_map, prob_agg)
    return rank_utils.aggregate_probabilities(prob_agg)

def assess_human_intervention_needed(
    recognized_class: dict,
    candidates: dict,
    confidence_score: float, 
    threshold: float = 0.5,
    additional_criteria: Optional[dict] = None
) -> Tuple[bool, str]:
    """
    Determine if human intervention is needed based on confidence score and other criteria.
    
    Args:
        confidence_score: Confidence score from judge_recognition_quality
        threshold: Threshold below which human intervention is needed
        additional_criteria: Optional dict with additional assessment criteria
    
    Returns:
        Tuple of (human_needed, reason)
    """
    reason = "Recognition quality acceptable"
    human_intervention_needed = False
    for class_name, class_value in recognized_class.items():
        candidate_name = class_name.replace("_class", "_candidates")
        if candidate_name in candidates:
            class_val_prob = candidates[candidate_name].get(class_value, 0.0) * confidence_score
            if class_val_prob <= threshold and threshold > 0:
                human_intervention_needed = True
                reason = f"Class '{class_name}' with value '{class_value}' has low confidence."


    return human_intervention_needed, reason