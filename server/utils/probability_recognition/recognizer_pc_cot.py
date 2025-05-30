import logging
import random
from math import ceil
from typing import Optional

import numpy as np

from .. import chat_restapi as llm
from .. import utils as ut
from . import rank_candidates as rank_utils

logger = logging.getLogger(__name__)


def rank_candidates_with_probs_cot(
    task_type: str,
    prompt_templates: dict,
    metadata: dict,
    candidate_list: list,
    other_classes: dict,
    interpretation: Optional[str] = None,
    model: Optional[str] = None,
    rounds: int = 1,
    user_config: Optional[dict] = None,
    complementary_knowledge: Optional[dict] = None,
    candidates_extraContent: Optional[dict] = None,
    max_items_to_show: int = 5,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
    use_tournament: Optional[bool] = None,
) -> dict:
    """
    Chain of Thought ranking: Uses two-stage approach with analysis then extraction.
    For large candidate sets, uses tournament-style ranking with probability redistribution.
    First stage uses default config for analysis, second stage uses logprob config for ranking.
    Returns a dict with candidate probabilities aggregated over multiple rounds.
    """
    if model is None:
        model = llm.DEFAULT_MODEL
    
    # Uncertain and OutOfSet are probability sink
    local_candidate_list = list(candidate_list)  # Create a local copy
    local_candidate_list = rank_utils.supplement_list(local_candidate_list, task_type)
    
    # Automatically decide whether to use tournament based on candidate count
    if use_tournament is None:
        use_tournament = len(local_candidate_list) > pool_size
    
    if use_tournament:
        logger.info(f"Using tournament CoT ranking for {len(local_candidate_list)} candidates")
        probs = _rank_candidates_tournament_cot(
            task_type, prompt_templates, metadata, local_candidate_list, other_classes,
            interpretation, model, rounds, user_config, complementary_knowledge,
            candidates_extraContent, max_items_to_show, pool_size, advance_ratio
        )
        logger.debug(f"Final tournament CoT probabilities: {probs}")
        return probs
    else:
        # Original CoT ranking logic for small candidate sets
        logger.debug(f"Using standard CoT ranking for {len(local_candidate_list)} candidates")
        return _rank_candidates_standard_cot(
            task_type, prompt_templates, metadata, local_candidate_list, other_classes,
            interpretation, model, rounds, user_config, complementary_knowledge,
            candidates_extraContent, max_items_to_show
        )
    

def _rank_candidates_standard_cot(
    task_type: str,
    prompt_templates: dict,
    metadata: dict,
    local_candidate_list: list,
    other_classes: dict,
    interpretation: Optional[str],
    model: str,
    rounds: int,
    user_config: Optional[dict],
    complementary_knowledge: Optional[dict],
    candidates_extraContent: Optional[dict],
    max_items_to_show: int,
) -> dict:
    """Standard CoT ranking logic for small candidate sets."""
    prob_agg: dict = {c: [] for c in local_candidate_list}
    
    # Get CoT prompt templates for ranking
    prompt_template_entry = prompt_templates[task_type]["stages"]
    prompt_template_analysis = prompt_template_entry["analysis"]
    prompt_template_extraction = prompt_template_entry["extraction"]

    kvPairs_analysis = prompt_templates[task_type].get("placeholders", [])
    kvPairs_analysis = {k: "" for k in kvPairs_analysis}

    # Prepare complementary knowledge string
    if complementary_knowledge is not None and "basic" in complementary_knowledge:
        complementary_knowledge_str = complementary_knowledge["basic"]
    else:
        complementary_knowledge_str = "No specific drilling terminology knowledge provided."
    
    # Convert other_classes dict to string representation
    if other_classes:
        other_classes_parts = []
        for class_type, class_value in other_classes.items():
            if isinstance(class_value, dict):
                # If it's a dictionary with probabilities, show top candidates with probabilities
                top_items = sorted(class_value.items(), key=lambda x: x[1], reverse=True)[:max_items_to_show]
                prob_str = ", ".join([f"{name}: {prob:.3f}" for name, prob in top_items])
                other_classes_parts.append(f"{class_type}: [{prob_str}]")
            else:
                # If it's a single value, show as before
                other_classes_parts.append(f"{class_type}: {class_value}")
        other_classes_str = "; ".join(other_classes_parts)
    else:
        other_classes_str = "Uncertain"
    
    for round_num in range(rounds):
        letter_map, reverse_map = ut.create_letter_mapping(local_candidate_list, model)
        
        # Create candidate list with letters for analysis
        if candidates_extraContent is not None:
            candidates_with_letters = []
            for cand, letter in letter_map.items():
                if cand in candidates_extraContent:
                    extra_content = candidates_extraContent[cand]
                    candidates_with_letters.append(f"{letter}: {cand}  Explanation: {extra_content}")
                else:
                    candidates_with_letters.append(f"{letter}: {cand}")
        else:
            candidates_with_letters = [f"{letter}: {cand}" for cand, letter in letter_map.items()]
        
        # Stage 1: Analysis with default config
        kvPairs_analysis.update({
            "<user_query>": str(metadata),
            "<candidates>": "\n".join(candidates_with_letters),
            "<recognized_class>": other_classes_str,
            "<interpretation>": interpretation or "",
            "<complementary_knowledge>": complementary_knowledge_str,
        })
        
        analysis_prompt = ut.assemble_prompt(prompt_template_analysis, kvPairs_analysis)
        analysis_result = ut.run_rag_task_single(analysis_prompt, model=model)  # Default config
        last_answer = analysis_result["content"] if analysis_result and "content" in analysis_result else ""
        
        logger.debug(f"Round {round_num + 1} analysis prompt: {analysis_prompt}")
        logger.debug(f"Round {round_num + 1} analysis result: {last_answer}")
        
        # Stage 2: Extraction with logprob config
        kvPairs_extraction = kvPairs_analysis.copy()
        kvPairs_extraction.update({"<last_answer>": last_answer})
        
        extraction_prompt = ut.assemble_prompt(prompt_template_extraction, kvPairs_extraction)
        config = rank_utils.prepare_llm_config(letter_map, model, user_config)  # Logprob config
        
        extraction_result = ut.run_rag_task_single(extraction_prompt, model=model, user_config=config)
        toplog_probs = rank_utils.extract_logprobs(extraction_result)
        
        logger.debug(f"Round {round_num + 1} extraction prompt: {extraction_prompt}")
        logger.debug(f"Round {round_num + 1} extraction result: {toplog_probs}")
        
        rank_utils.update_prob_aggregation(toplog_probs, reverse_map, prob_agg)
        logger.debug(f"Round {round_num + 1} prob_agg: {prob_agg}")

    return rank_utils.aggregate_probabilities(prob_agg)


def _rank_candidates_tournament_cot(
    task_type: str,
    prompt_templates: dict,
    metadata: dict,
    candidate_list: list,
    other_classes: dict,
    interpretation: Optional[str],
    model: str,
    rounds: int,
    user_config: Optional[dict],
    complementary_knowledge: Optional[dict],
    candidates_extraContent: Optional[dict],
    max_items_to_show: int,
    pool_size: int,
    advance_ratio: float,
) -> dict:
    """
    Tournament-style CoT ranking with probability redistribution.
    
    Args:
        pool_size: Maximum candidates per pool (default: 6)
        advance_ratio: Fraction of candidates advancing (default: 1/3)
    
    Returns:
        dict: Final candidate probabilities after tournament
    """
    
    # Track all candidates with initial equal probability
    all_candidates_probs = {c: 1.0 / len(candidate_list) for c in candidate_list}
    active_candidates = candidate_list.copy()
    
    tournament_round = 1
    carry_over_ratio = 1.0  # Initialize carry-over ratio
    
    # Tournament rounds
    while len(active_candidates) > pool_size:
        logger.info(f"Tournament CoT Round {tournament_round}: {len(active_candidates)} candidates")
        
        # Create pools
        pools = _create_tournament_pools(active_candidates, pool_size)
        
        # Rank each pool and collect results
        round_winners = []
        round_winner_probs = {}
        round_loser_probs = {}
        
        for pool_idx, pool in enumerate(pools):
            logger.debug(f"Ranking CoT pool {pool_idx + 1}/{len(pools)} with {len(pool)} candidates")
            
            # Rank this pool using CoT
            pool_probs = _rank_single_pool_cot(
                pool, task_type, prompt_templates, metadata, other_classes,
                interpretation, model, rounds, user_config, 
                complementary_knowledge, candidates_extraContent, max_items_to_show
            )
            
            # Select winners and losers
            pool_winners, pool_winner_probs_dict, pool_loser_probs_dict = _select_pool_winners(
                pool_probs, advance_ratio
            )
            
            round_winners.extend(pool_winners)
            round_winner_probs.update(pool_winner_probs_dict)
            round_loser_probs.update(pool_loser_probs_dict)
        
        # Redistribute probabilities
        all_candidates_probs, carry_over_ratio = _redistribute_tournament_probabilities(
            all_candidates_probs, round_winner_probs, round_loser_probs, len(pools), carry_over_ratio
        )
        
        # Prepare for next round
        active_candidates = round_winners
        tournament_round += 1
    
    # Final round
    if len(active_candidates) > 1:
        logger.info(f"Final CoT Round: {len(active_candidates)} candidates")
        
        final_probs = _rank_single_pool_cot(
            active_candidates, task_type, prompt_templates, metadata, other_classes,
            interpretation, model, rounds, user_config, 
            complementary_knowledge, candidates_extraContent, max_items_to_show
        )
        
        # Final probability redistribution
        all_candidates_probs = _apply_final_redistribution(
            all_candidates_probs, final_probs, active_candidates, carry_over_ratio
        )
    
    return all_candidates_probs


def _rank_single_pool_cot(
    pool_candidates: list,
    task_type: str,
    prompt_templates: dict,
    metadata: dict,
    other_classes: dict,
    interpretation: Optional[str],
    model: str,
    rounds: int,
    user_config: Optional[dict],
    complementary_knowledge: Optional[dict],
    candidates_extraContent: Optional[dict],
    max_items_to_show: int,
) -> dict:
    """Rank candidates within a single pool using CoT logic."""
    # Use existing CoT ranking logic
    return _rank_candidates_standard_cot(
        task_type, prompt_templates, metadata, pool_candidates, other_classes,
        interpretation, model, rounds, user_config, complementary_knowledge,
        candidates_extraContent, max_items_to_show
    )


# Copy tournament helper functions from recognizer_pc.py
def _create_tournament_pools(candidates: list, pool_size: int) -> list:
    """Split candidates into roughly equal pools with maximum size of pool_size."""
    shuffled = candidates.copy()
    random.shuffle(shuffled)  # Randomize to avoid bias
    
    num_candidates = len(shuffled)
    
    # Calculate number of pools needed to keep pool sizes reasonable
    num_pools = ceil(num_candidates / pool_size)
    
    # Use numpy's array_split for even distribution
    pools = np.array_split(shuffled, num_pools)
    
    # Convert back to lists
    return [pool.tolist() for pool in pools]


def _select_pool_winners(pool_probs: dict, advance_ratio: float) -> tuple:
    """Select winners and losers from pool ranking."""
    sorted_candidates = sorted(pool_probs.items(), key=lambda x: x[1], reverse=True)
    num_winners = max(1, int(len(sorted_candidates) * advance_ratio))
    
    winners = [name for name, prob in sorted_candidates[:num_winners]]
    winner_probs = {name: prob for name, prob in sorted_candidates[:num_winners]}
    loser_probs = {name: prob for name, prob in sorted_candidates[num_winners:]}
    
    return winners, winner_probs, loser_probs


def _redistribute_tournament_probabilities(
    all_probs: dict,
    round_winner_probs: dict,
    round_loser_probs: dict,
    num_pools: int,
    carry_over_ratio: float = 1.0,
) -> tuple:
    """
    Redistribute probabilities based on tournament performance.
    
    Returns:
        tuple: (updated_probs, new_carry_over_ratio)
    """
    # Calculate sums
    winner_sum = sum(round_winner_probs.values())
    loser_sum = sum(round_loser_probs.values())
    
    # Calculate new carry-over ratio based on performance
    # Mandated minimum so this round losers are above previous round losers
    mandated_minimum = (1 - carry_over_ratio) / len(round_loser_probs) if len(round_loser_probs) > 0 else 0
    # This represents how much probability the winners collectively earned
    new_carry_over_ratio = carry_over_ratio * (winner_sum / num_pools) + (carry_over_ratio - 1) if num_pools > 0 else carry_over_ratio
    
    logger.debug(f"Tournament CoT redistribution: winner_sum={winner_sum:.3f}, loser_sum={loser_sum:.3f}, "
                f"input_carry_over={carry_over_ratio:.3f}, new_carry_over={new_carry_over_ratio:.3f}")
    
    # Update probabilities
    updated_probs = all_probs.copy()
    
    # Winners get their proportional share of the carry-over probability
    if winner_sum > 0:
        for winner, prob in round_winner_probs.items():
            updated_probs[winner] = carry_over_ratio * (prob / winner_sum)
    
    # Losers split the remaining probability proportionally
    loser_share = carry_over_ratio - carry_over_ratio * (winner_sum / num_pools)
    if loser_sum > 0 and loser_share > 0:
        for loser, prob in round_loser_probs.items():
            updated_probs[loser] = mandated_minimum + loser_share * (prob / loser_sum)
    
    return updated_probs, new_carry_over_ratio


def _apply_final_redistribution(
    all_probs: dict,
    final_probs: dict,
    active_candidates: list,
    carry_over_ratio: float,
) -> dict:
    """Apply final round results to overall probability distribution."""
    # Redistribute based on final ranking using the accumulated carry-over ratio
    updated_probs = all_probs.copy()
    final_sum = sum(final_probs.values())
    # Mandated minimum so the final round candidates get a minimum share
    mandated_minimum = (1 - carry_over_ratio) / len(active_candidates) if len(active_candidates) > 0 else 0
    # New carry-over ratio for final redistribution
    new_carry_over_ratio = 2 * carry_over_ratio - 1 if len(active_candidates) > 0 else carry_over_ratio
    
    if final_sum > 0:
        for candidate in active_candidates:
            if candidate in final_probs:
                updated_probs[candidate] = mandated_minimum + new_carry_over_ratio * (final_probs[candidate] / final_sum)

    return updated_probs


def _rank_quantity_candidates_cot(
    user_query: dict,
    interpretation: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    number_of_candidates: dict,
    models_high_low: list,
    rounds: int,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
) -> tuple:
    """
    Rank quantity candidates using CoT approach and return the best candidate with all probabilities.
    
    Returns:
        tuple: (quantity_class, quantity_probs)
    """
    quantity_fullList = list(quantity_fullList_extraContent.keys())
    
    quantity_candidates = quantity_fullList
    quantity_candidates_extraContent = ut.supplement_quantity_candidates(
        quantity_fullList, [], quantity_fullList_extraContent
    )
    
    quantity_probs = rank_candidates_with_probs_cot(
        task_type="Rank_quantity",
        prompt_templates=prompt_templates,
        metadata=user_query,
        candidate_list=quantity_candidates,
        other_classes={},
        interpretation=interpretation,
        model=models_high_low[1],
        rounds=rounds,
        complementary_knowledge=complementary_knowledge,
        candidates_extraContent=quantity_candidates_extraContent,
        pool_size=pool_size,
        advance_ratio=advance_ratio,
        use_tournament=None,  # Auto-decide based on candidate count
    )
    
    # Get the most probable quantity class
    if quantity_probs:
        quantity_class = max(quantity_probs.keys(), key=lambda k: quantity_probs[k])
        logger.info(f"quantity_class: {quantity_class}")
        quantity_probs = dict(sorted(quantity_probs.items(), key=lambda x: x[1], reverse=True)[:number_of_candidates.get("Quantity_class", 10)])
    else:
        quantity_class = "UncertainQuantity"

    return quantity_class, quantity_probs


def _rank_unit_candidates_cot(
    user_query: dict,
    interpretation: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    unit_fullList_extraContent: dict,
    quantity_probs: dict,
    quantity_extra_content: dict,
    number_of_candidates: dict,
    models_high_low: list,
    rounds: int,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
) -> tuple:
    """
    Rank unit candidates based on top quantity candidates using CoT approach.
    
    Returns:
        tuple: (unit_class, unit_probs)
    """
    # Get units related to top quantity candidates
    top_quantity_names = list(quantity_probs.keys())

    unit_candidates_related = ut.retrieve_unit_relatedTo_Quantiy(top_quantity_names)
    
    # Flatten the related units
    unit_candidates_list = list(unit_candidates_related.keys())
    
    # Use full list if no related units found
    if not unit_candidates_list:
        unit_candidates_list = list(unit_fullList_extraContent.keys())
    
    # Get other classes for unit ranking - pass quantity probabilities instead of single class
    other_classes_unit = {"Quantity_class": quantity_probs}

    # Prepare extra content for unit candidates
    if unit_fullList_extraContent is not None and unit_candidates_list:
        unit_candidates_extraContent = ut.supplement_unit_candidates(
            unit_candidates_list, [], unit_fullList_extraContent
        )
    else:
        unit_candidates_extraContent = unit_candidates_related
    
    # Rank unit candidates
    unit_probs = rank_candidates_with_probs_cot(
        task_type="Rank_unit",
        prompt_templates=prompt_templates,
        metadata=user_query,
        candidate_list=unit_candidates_list,
        other_classes=other_classes_unit,
        interpretation=interpretation,
        model=models_high_low[1],
        rounds=rounds,
        complementary_knowledge=complementary_knowledge,
        candidates_extraContent=unit_candidates_extraContent,
        max_items_to_show=number_of_candidates.get("Quantity_class", 5),
        pool_size=pool_size,
        advance_ratio=advance_ratio,
        use_tournament=None,  # Auto-decide based on candidate count
    )
    
    # Get the most probable unit class
    if unit_probs:
        unit_class = max(unit_probs.keys(), key=lambda k: unit_probs[k])
        logger.info(f"unit_class: {unit_class}")
        unit_probs = dict(sorted(unit_probs.items(), key=lambda x: x[1], reverse=True)[:number_of_candidates.get("Unit_class", 10)])
    else:
        unit_class = "UncertainUnit"
    
    return unit_class, unit_probs


def _rank_prototypeData_candidates_cot(
    user_query: dict,
    interpretation: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    prototypeData_fullList_extraContent: dict,
    quantity_probs: dict,
    quantity_extra_content: dict,
    number_of_candidates: dict,
    models_high_low: list,
    rounds: int,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
) -> tuple:
    """
    Rank prototypeData candidates based on top quantity candidates using CoT approach.
    
    Returns:
        tuple: (prototypeData_class, prototypeData_probs)
    """
    # Get prototypeData related to top quantity candidates
    top_quantity_names = list(quantity_probs.keys())

    prototypeData_candidates_related = ut.retrieve_prototypeData_relatedTo_Quantiy(top_quantity_names)

    # Flatten the related prototypeData
    prototypeData_candidates_list = list(prototypeData_candidates_related.keys())
    
    # use full list if no related prototypeData found
    if not prototypeData_candidates_list:
        prototypeData_candidates_list = list(prototypeData_fullList_extraContent.keys())
    
    # Get other classes for prototypeData ranking - pass probabilities instead of single classes
    other_classes_prototype = {
        "Quantity_class": quantity_probs,
    }
    
    # Prepare extra content for prototypeData candidates
    if prototypeData_fullList_extraContent is not None and prototypeData_candidates_list:
        prototypeData_candidates_extraContent = ut.supplement_prototypeData_candidates(
            prototypeData_candidates_list, [], prototypeData_fullList_extraContent
        )
    else:
        prototypeData_candidates_extraContent = prototypeData_candidates_related
    
    # Rank prototypeData candidates
    prototypeData_probs = rank_candidates_with_probs_cot(
        task_type="Rank_prototypeData",
        prompt_templates=prompt_templates,
        metadata=user_query,
        candidate_list=prototypeData_candidates_list,
        other_classes=other_classes_prototype,
        interpretation=interpretation,
        model=models_high_low[0],
        rounds=rounds,
        complementary_knowledge=complementary_knowledge,
        candidates_extraContent=prototypeData_candidates_extraContent,
        max_items_to_show=number_of_candidates.get("Quantity_class", 5),
        pool_size=pool_size,
        advance_ratio=advance_ratio,
        use_tournament=None,  # Auto-decide based on candidate count
    )
    
    # Get the most probable prototypeData class
    if prototypeData_probs:
        prototypeData_class = max(prototypeData_probs.keys(), key=lambda k: prototypeData_probs[k])
        logger.info(f"prototypeData_class: {prototypeData_class}")
        prototypeData_probs = dict(sorted(prototypeData_probs.items(), key=lambda x: x[1], reverse=True)[:number_of_candidates.get("PrototypeData_class", 5)])
    else:
        prototypeData_class = "UncertainPrototypeData"

    return prototypeData_class, prototypeData_probs


def recoginize_metadata(
    user_query: dict,
    interpretation: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    unit_fullList_extraContent: dict,
    prototypeData_fullList_extraContent: dict,
    models_high_low: list = [llm.LOW_MODEL, llm.HIGH_MODEL],
    number_of_candidates: Optional[dict] = None,
    rounds: int = 1,
    pool_size: int = 6,
    advance_ratio: float = 1/3,
) -> tuple:
    """
    Chain of Thought metadata recognition with probabilistic ranking.
    Uses two-stage CoT approach: analysis then extraction with logprobs.
    Returns (recognized_class, candidates_with_probs, prompts)
    
    Args:
        number_of_candidates: Dict specifying max candidates to return for each type
            e.g., {"Quantity_class": 5, "Unit_class": 10, "PrototypeData_class": 5}
        pool_size: Maximum candidates per pool for tournament ranking
        advance_ratio: Fraction of candidates advancing in tournament
    
    Returns:
        tuple: (recognized_class, candidates_with_probs, prompts)
            - recognized_class: Dict with most probable class for each type
            - candidates_with_probs: Dict with top N candidates and their probabilities for each type
            - prompts: Dict with ranking information and full probability data
    """
    # Set default candidate numbers if not provided
    if number_of_candidates is None:
        number_of_candidates = {
            "Quantity_class": 5,
            "Unit_class": 10,
            "PrototypeData_class": 5,
        }

    # Results to return
    recognized_class = {}
    candidates_with_probs = {}
    prompts = {}

    # === Step 1: Rank Quantity Candidates ===
    quantity_class, quantity_probs = _rank_quantity_candidates_cot(
        user_query=user_query,
        interpretation=interpretation,
        complementary_knowledge=complementary_knowledge,
        prompt_templates=prompt_templates,
        quantity_fullList_extraContent=quantity_fullList_extraContent,
        number_of_candidates=number_of_candidates,
        models_high_low=models_high_low,
        rounds=rounds,
        pool_size=pool_size,
        advance_ratio=advance_ratio,
    )
    
    recognized_class["Quantity_class"] = quantity_class
    candidates_with_probs["Quantity_candidates"] = quantity_probs

    # === Step 2: Rank Unit Candidates ===
    unit_class, unit_probs = _rank_unit_candidates_cot(
        user_query=user_query,
        interpretation=interpretation,
        complementary_knowledge=complementary_knowledge,
        prompt_templates=prompt_templates,
        unit_fullList_extraContent=unit_fullList_extraContent,
        quantity_probs=quantity_probs,
        quantity_extra_content=quantity_fullList_extraContent,
        number_of_candidates=number_of_candidates,
        models_high_low=models_high_low,
        rounds=rounds,
        pool_size=pool_size,
        advance_ratio=advance_ratio,
    )
    
    recognized_class["Unit_class"] = unit_class
    candidates_with_probs["Unit_candidates"] = unit_probs

    # === Step 3: Rank PrototypeData Candidates ===
    prototypeData_class, prototypeData_probs = _rank_prototypeData_candidates_cot(
        user_query=user_query,
        interpretation=interpretation,
        complementary_knowledge=complementary_knowledge,
        prompt_templates=prompt_templates,
        prototypeData_fullList_extraContent=prototypeData_fullList_extraContent,
        quantity_probs=quantity_probs,
        quantity_extra_content=quantity_fullList_extraContent,
        number_of_candidates=number_of_candidates,
        models_high_low=models_high_low,
        rounds=rounds,
        pool_size=pool_size,
        advance_ratio=advance_ratio,
    )
    
    recognized_class["PrototypeData_class"] = prototypeData_class
    candidates_with_probs["PrototypeData_candidates"] = prototypeData_probs

    # === Step 4: Retrieve MeasurableQuantity class ===
    MQuantity_class = prototypeData_fullList_extraContent[prototypeData_class].get("ddhub:IsOfMeasurableQuantity", "UncertainMeasurableQuantity")
    recognized_class["MeasurableQuantity_class"] = MQuantity_class


    # === Finalize Candidates with Probabilities ===
    # Update candidates_with_probs with filtered results
    candidates_with_probs = {
        "Quantity_candidates": quantity_probs,
        "Unit_candidates": unit_probs,
        "PrototypeData_candidates": prototypeData_probs,
    }

    # Store ranking prompts (simplified for now)
    prompts = {
        "prompt_rank_quantity": f"Ranked {len(quantity_fullList_extraContent)} quantity candidates",
        "prompt_rank_unit": f"Ranked unit candidates", 
        "prompt_rank_prototypeData": f"Ranked prototypeData candidates",
    }

    toReturn = (recognized_class, candidates_with_probs, prompts)
    return toReturn

def interpret_mnemonic(
    user_query: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
    """Chain of Thought interpretation with analysis and extraction stages."""
    prompt_template_entry = prompt_templates["Interpret_mnemonic"]["stages"]
    prompt_template_analysis = prompt_template_entry["analysis"]
    prompt_template_extraction = prompt_template_entry["extraction"]

    kvPairs = prompt_templates["Interpret_mnemonic"].get("placeholders", [])
    kvPairs = {k: "" for k in kvPairs}

    if complementary_knowledge is not None:
        complementary_knowledge_str = complementary_knowledge["basic"]
    else:
        complementary_knowledge_str = ""

    if user_query is None:
        logger.error("user_query is not provided!")
        raise ValueError("user_query is not provided!")
    user_query_str = user_query if isinstance(user_query, str) else str(user_query)

    # Stage 1: Analysis with default config
    kvPairs.update({
        "<user_query>": user_query_str,
        "<complementary_knowledge>": complementary_knowledge_str,
    })
    analysis_prompt = ut.assemble_prompt(prompt_template_analysis, kvPairs)
    analysis_result = ut.run_rag_task_single(analysis_prompt, model=model)  # Default config

    logger.info(f"Interpret_mnemonic analysis prompt:\n{analysis_prompt}")

    # Stage 2: Extraction with default config (no logprobs needed for interpretation)
    kvPairs.update({"<last_answer>": analysis_result["content"] if analysis_result else ""})
    extraction_prompt = ut.assemble_prompt(prompt_template_extraction, kvPairs)
    extraction_result = ut.run_rag_task_single(extraction_prompt, model=model)  # Default config

    interpretation = str(extraction_result["content"]) if extraction_result else ""

    logger.info(f"Interpret_mnemonic extraction prompt:\n{extraction_prompt}")
    logger.info(f"Interpret_mnemonic extraction result:\n{interpretation}")

    return interpretation, extraction_prompt

