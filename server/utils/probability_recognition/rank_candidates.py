import copy
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional, Union

sys.path.append(str(Path(__file__).resolve().parent))

from .. import chat_restapi as llm
from .. import utils as ut

logger = logging.getLogger(__name__)


def supplement_list(input_list, task_type):
    if "quantity" in task_type.lower():
        return input_list + ["UncertainQuantity", "OutOfSetQuantity"]
    if "unit" in task_type.lower():
        return input_list + ["UncertainUnit", "OutOfSetUnit"]
    if "prototype" in task_type.lower():
        return input_list + ["UncertainPrototypeData", "OutOfSetPrototypeData"]
    return input_list


def _build_ranking_prompt(metadata, letter_map, other_classes, interpretation, prompt_templates, task_type, complementary_knowledge=None):
    """Build the prompt for ranking candidates."""
    template = prompt_templates[task_type]["ranking"]

    kvPairs = {k: "" for k in prompt_templates[task_type].get("placeholders", [])}

    # Create candidate list with letters for template
    candidates_with_letters = [f"{letter}: {cand}" for cand, letter in letter_map.items()]

    # Prepare complementary knowledge string
    if complementary_knowledge is not None and "basic" in complementary_knowledge:
        complementary_knowledge_str = complementary_knowledge["basic"]
    else:
        complementary_knowledge_str = "No specific drilling terminology knowledge provided."

    kvPairs.update({
        "<user_query>": metadata,
        "<candidates>": "\n".join(candidates_with_letters),
        "<recognized_class>": other_classes,
        "<interpretation>": interpretation or "",
        "<complementary_knowledge>": complementary_knowledge_str,
    })
    return ut.assemble_prompt(template, kvPairs)


def prepare_llm_config(letter_map, model, user_config=None):
    """Prepare the configuration for the LLM call."""
    config = user_config.copy() if user_config else {}
    config.update(
        {
            "logprobs": True,
            "temperature": 0.0,  # Use deterministic sampling
            "top_p": 1.0,  # Use full probability distribution
            "top_logprobs": min(20, len(letter_map)),
            "logit_bias": {ut.get_token_id(letter, model=model)[0]: 100 for letter in letter_map.values()},
            "max_completion_tokens": 1,
        }
    )
    return config


def extract_logprobs(result):
    """Extract and flatten logprobs from LLM result."""
    logprobs_list = result.get("logprobs", [])
    toplog_probs = []
    for x in logprobs_list:
        toplog_probs.extend(x.get("top_logprobs", []))
    return toplog_probs


def update_prob_aggregation(toplog_probs, reverse_map, prob_agg):
    """Update probability aggregation with logprobs from one round."""
    round_probs = {x: 0.0 for x in prob_agg.keys()}
    for logprob_dict in toplog_probs:
        token = logprob_dict["token"]
        logprob = logprob_dict["logprob"]
        if token.strip() in reverse_map:
            round_probs[reverse_map[token.strip()]] += math.exp(logprob)
    # Normalize the probabilities for this round
    total_prob = sum(round_probs.values())
    round_probs = {k: v / total_prob if total_prob > 0 else 0 for k, v in round_probs.items()}
    # Append the aggregated probabilities for this round
    for key in round_probs:
        prob_agg[key].append(round_probs[key])


def aggregate_probabilities(prob_agg):
    """Aggregate probabilities across all rounds."""
    prob_summary = {}
    for cand, probs in prob_agg.items():
        valid_probs = [p for p in probs if p is not None]
        if valid_probs:
            prob_summary[cand] = sum(valid_probs) / len(valid_probs)
        else:
            prob_summary[cand] = 0.0
    # Normalize the probabilities
    total_prob = sum(prob_summary.values())
    if total_prob > 0:
        prob_summary = {k: v / total_prob for k, v in prob_summary.items()}
    else:
        # If all probabilities are zero, return them as is
        prob_summary = {k: 0.0 for k in prob_summary.keys()}
    return prob_summary


def rank_candidates_with_probs(
    task_type: str,
    prompt_templates: dict,
    metadata: Union[dict, str],
    candidate_list: list,
    other_classes: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 5,
    user_config: Optional[dict] = None,
    complementary_knowledge: Optional[dict] = None,
) -> dict:
    """
    Rank candidates using LLM with logprob extraction and candidate shuffling.
    Returns a dict with candidate probabilities aggregated over multiple rounds.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    # Uncertain and OutOfSet are probability sink
    local_candidate_list = list(candidate_list)  # Create a local copy

    prob_agg: dict = {c: [] for c in local_candidate_list}

    for round_num in range(rounds):
        shuffled_candidate_list = copy.deepcopy(local_candidate_list)
        random.shuffle(shuffled_candidate_list)
        letter_map, reverse_map = ut.create_letter_mapping(shuffled_candidate_list)
        prompt = _build_ranking_prompt(metadata, letter_map, other_classes, interpretation, prompt_templates, task_type, complementary_knowledge)
        logger.debug(f"Round {round_num + 1} prompt: {prompt}")
        config = prepare_llm_config(letter_map, model, user_config)

        result = ut.run_rag_task_single(prompt, model=model, user_config=config)
        toplog_probs = extract_logprobs(result)

        logger.debug(f"Round {round_num + 1} result: {toplog_probs}")
        update_prob_aggregation(toplog_probs, reverse_map, prob_agg)
        logger.debug(f"Round {round_num + 1} prob_agg: {prob_agg}")

    return aggregate_probabilities(prob_agg)
    # return prob_agg


def test_rank_candidates_with_probs():
    """
    Simple test for rank_candidates_with_probs: checks that output keys match input candidates + 'None', and values are floats.
    """
    prompt_templates = {
        "Rank": {
            "ranking": "Task Objective: Given a user query and a list of candidate classes, select the single most likely candidate and respond with only its corresponding letter.\nInput:\nUser query: <user_query>\nCandidate list:\n<candidates>\nInterpretation: <interpretation>\nComplementary knowledge: <complementary_knowledge>\nOutput: (Only the single letter of the most probable candidate)"
        }
    }
    user_query = {"Mnemonic": "HKLO", "Description": "Hookload", "Unit": "kkgf", "DataType": "double"}
    other_classes = {"Unit_class": "BarPerMinute", "PrototypeData_class": "None"}
    candidate_list = ["ForceQuantity", "PressureQuantity", "None"]
    complementary_knowledge = {"basic": "HKLD: Hookload. HKD: Hookload. kkgf: kilo kilogram force, or thousand kilogram force, or ton force."}
    # Use a dummy model name; in real test, mock ut.run_rag_task_single if needed
    try:
        result = rank_candidates_with_probs(
            task_type="Rank",
            prompt_templates=prompt_templates,
            metadata=user_query,
            candidate_list=candidate_list,
            other_classes=other_classes,
            interpretation="HKLO means Hookload",
            model="gpt-4o-mini",
            rounds=5,
            complementary_knowledge=complementary_knowledge,
        )
    except Exception as e:
        print("Test failed to run LLM call (expected if no API):", e)
        return
    assert set(result.keys()) == set(candidate_list), f"Output keys: {result.keys()}"
    for v in result.values():
        assert isinstance(v, float), f"Output value is not float: {v}"
    logger.debug(f"Ranking result: {result}")
    print("test_rank_candidates_with_probs passed.")


def rank_all_candidates(
    candidates: dict,
    recognized_class: dict,
    prompt_templates: dict,
    metadata: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 5,
    user_config: Optional[dict] = None,
    complementary_knowledge: Optional[dict] = None,
) -> dict:
    """
    Rank all candidate types (Quantity, Unit, PrototypeData) and return their probabilities.

    Args:
        candidates: Dict containing candidate lists from recognize_metadata
        recognized_class: Dict with recognized classes for other types
        prompt_templates: Prompt templates for ranking
        metadata: User query metadata
        interpretation: Optional interpretation string
        model: LLM model to use
        rounds: Number of rounds for probability aggregation
        user_config: Optional LLM configuration
        complementary_knowledge: Optional drilling terminology knowledge

    Returns:
        Dict with `candidates` key containing ranked candidates for each type.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    result = {}

    # Rank Quantity candidates
    quantity_candidates = candidates.get("Quantity_candidates", [])
    other_classes_q = {k: v for k, v in recognized_class.items() if k != "Quantity_class"}
    if quantity_candidates:
        local_candidate_list = supplement_list(list(quantity_candidates.keys()), "quantity")
        result["Quantity_candidates"] = rank_candidates_with_probs(
            task_type="Rank",
            prompt_templates=prompt_templates,
            metadata=metadata,
            candidate_list=local_candidate_list,
            other_classes=other_classes_q,
            interpretation=interpretation,
            model=model,
            rounds=rounds,
            user_config=user_config,
            complementary_knowledge=complementary_knowledge,
        )

    # Rank Unit candidates
    unit_candidates = candidates.get("Unit_candidates", [])
    other_classes_u = {k: v for k, v in recognized_class.items() if k != "Unit_class"}
    if unit_candidates:
        local_candidate_list = supplement_list(list(unit_candidates.keys()), "unit")
        result["Unit_candidates"] = rank_candidates_with_probs(
            task_type="Rank",
            prompt_templates=prompt_templates,
            metadata=metadata,
            candidate_list=local_candidate_list,
            other_classes=other_classes_u,
            interpretation=interpretation,
            model=model,
            rounds=rounds,
            user_config=user_config,
            complementary_knowledge=complementary_knowledge,
        )

    # Rank PrototypeData candidates
    prototypedata_candidates = candidates.get("PrototypeData_candidates", [])
    other_classes_p = {k: v for k, v in recognized_class.items() if k != "PrototypeData_class"}
    if prototypedata_candidates:
        local_candidate_list = supplement_list(list(prototypedata_candidates.keys()), "prototype_data")
        result["PrototypeData_candidates"] = rank_candidates_with_probs(
            task_type="Rank",
            prompt_templates=prompt_templates,
            metadata=metadata,
            candidate_list=local_candidate_list,
            other_classes=other_classes_p,
            interpretation=interpretation,
            model=model,
            rounds=rounds,
            user_config=user_config,
            complementary_knowledge=complementary_knowledge,
        )

    return {"candidates": result}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_rank_candidates_with_probs()
