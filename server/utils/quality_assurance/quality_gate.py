import logging
from typing import Optional, Union

import numpy as np

from .. import chat_restapi as llm
from .. import sparql_connector as sc  # noqa
from .. import utils as ut
from ..configs import globals_config as glb
from ..configs import log_config  # noqa F401
from ..validation import probability_based_selection as pbs
from ..quality_assurance import BM25_predicator as bm25
from ..quality_assurance import embedding_predictor as embedding

# from . import rank_candidates as rk


logger = logging.getLogger(__name__)
prompt_template_collection = glb.prompt_template_collection
complementary_knowledge = glb.complementary_knowledge
quantity_fullList_extraContent = glb.quantity_fullList_extraContent
unit_fullList_extraContent = glb.unit_fullList_extraContent
prototypeData_fullList_extraContent = glb.prototypeData_fullList_extraContent


def validate_ontology_consistency(quantity: str, unit: str, prototypedata: str) -> bool:
    """
    Validates the consistency between a given quantity, unit, and prototypedata based on
    predefined ontology rules stored in the knowledge base.

    This function performs two main checks:
    1. Ensures that the provided `quantity` is valid for the given `prototypedata`.
    2. Ensures that the provided `unit` is valid for the given `quantity`.

    Args:
        quantity (str): The quantity class to validate (e.g., "ForceQuantity").
        unit (str): The unit class to validate (e.g., "KilogramForce").
        prototypedata (str): The prototype data class to validate (e.g., "HookLoad").

    Returns:
        bool: True if the quantity, unit, and prototypedata are consistent according to
              the ontology; False otherwise.
    """
    # Log inputs and check for None values, which indicate invalid or missing recognition
    if quantity is None or unit is None or prototypedata is None:
        logger.error(f"Input contains None. Quantity: {quantity}, Unit: {unit}, PrototypeData: {prototypedata}. Returning False.")
        return False

    # Query the knowledge base for valid quantities associated with the prototypedata
    valid_quantities_for_prototypedata = glb.prototypeData_fullList_extraContent.get(prototypedata, {}).get("ddhub:IsOfBaseQuantity", [])
    logger.info(f"Valid quantities for '{prototypedata}': {valid_quantities_for_prototypedata}")
    # Check if the provided quantity is in the list of valid quantities for the prototypedata
    if quantity not in valid_quantities_for_prototypedata:
        logger.error(f"Quantity '{quantity}' not found in valid quantities for '{prototypedata}'. Returning False.")
        return False

    # Query the knowledge base for valid units associated with the quantity
    valid_units_for_quantity = glb.quantity_fullList_extraContent.get(quantity, {}).get("zzz:QuantityHasUnit", [])
    logger.info(f"Valid units for '{quantity}': {valid_units_for_quantity}")
    # Check if the provided unit is in the list of valid units for the quantity
    if unit not in valid_units_for_quantity:
        logger.error(f"Unit '{unit}' not found in valid units for '{quantity}'. Returning False.")
        return False

    # If all checks pass, the ontology is consistent
    logger.info(f"All ontology consistency checks passed for Quantity='{quantity}', Unit='{unit}', PrototypeData='{prototypedata}'.")
    return True


# def contains_no_none_recognition(quantity: str, unit: str, prototypeData: str) -> bool:
#     """
#     Checks if any of the input parameters (quantity, unit, prototypeData) are explicitly
#     the string "None". This is used to identify cases where the recognition process
#     explicitly returned "None" as a class, indicating an invalid or unrecognized value.

#     Args:
#         quantity (str): The recognized quantity class.
#         unit (str): The recognized unit class.
#         prototypeData (str): The recognized prototype data class.

#     Returns:
#         bool: True if none of the input parameters are the string "None"; False otherwise.
#     """
#     # Check if any of the input strings are "None"
#     if quantity == "None" or unit == "None" or prototypeData == "None":
#         logger.info(f"One or more inputs are 'None' string. Quantity: {quantity}, Unit: {unit}, PrototypeData: {prototypeData}. Returning False.")
#         return False
#     logger.info(f"No 'None' string inputs found. Quantity: {quantity}, Unit: {unit}, PrototypeData: {prototypeData}. Returning True.")
#     return True


def _get_sorted_valid_candidates(candidates_with_probabilities: list, min_probability_cutoff: float) -> list:
    """
    Filters and sorts candidates based on a minimal probability cutoff.

    This helper function processes a list of dictionaries, where each dictionary
    represents a candidate and its aggregated probabilities (average and history).
    It filters out candidates below a `min_probability_cutoff` and then sorts
    the remaining candidates by their average probability in descending order.

    Args:
        candidates_with_probabilities (list): A list of dictionaries, where each dictionary
                                              contains an option and its aggregated data.
                                              Example: [{"HookLoad": {"average": 0.805, "history": [0.8, 0.81]}}, ...]
        min_probability_cutoff (float): The minimal average probability for a candidate to be considered.
                                        Candidates with an average probability below this will be dropped.

    Returns:
        list: A list of dictionaries, each containing "name" and "probability" for valid candidates,
              sorted by probability in descending order. Returns an empty list if no candidates
              meet the criteria or if the input is empty.
    """
    if not candidates_with_probabilities:
        logger.debug("No candidates with probabilities provided to _get_sorted_valid_candidates. Returning empty list.")
        return []

    processed_candidates = []
    for candidate_data in candidates_with_probabilities:
        # Each item in the list is a dictionary with a single key-value pair
        candidate_name = list(candidate_data.keys())[0]
        avg_probability = candidate_data[candidate_name]["average"]

        # Apply the minimal probability cutoff
        if avg_probability >= min_probability_cutoff:
            processed_candidates.append({"name": candidate_name, "probability": avg_probability})

    # Sort the remaining candidates by their average probability in descending order
    sorted_candidates = sorted(processed_candidates, key=lambda x: x.get("probability", 0), reverse=True)
    logger.debug(f"Sorted valid candidates: {sorted_candidates}")
    return sorted_candidates


# def _is_top_candidate_above_threshold(recognized_candidate: str, candidates_with_probabilities: list, min_probability_cutoff: float, threshold: float) -> bool:
#     """
#     Checks if the recognized candidate is the top candidate among those above a minimal
#     probability cutoff, and if its average probability is above a specified threshold.

#     Args:
#         recognized_candidate (str): The specific candidate that was recognized as the primary choice.
#         candidates_with_probabilities (list): A list of dictionaries, where each dictionary
#                                               contains an option and its aggregated data.
#                                               Example: [{"HookLoad": {"average": 0.805, "history": [0.8, 0.81]}}, ...]
#         min_probability_cutoff (float): The minimal average probability for a candidate to be considered.
#                                         Candidates with an average probability below this will be dropped.
#         threshold (float): The minimum average probability for the top remaining candidate to be
#                            considered valid.

#     Returns:
#         bool: True if the recognized candidate is the top one and its average probability
#               is above the threshold, False otherwise.
#     """
#     # Get candidates that meet the minimal probability cutoff, sorted by probability
#     sorted_valid_candidates = _get_sorted_valid_candidates(candidates_with_probabilities, min_probability_cutoff)

#     if not sorted_valid_candidates:
#         logger.info(f"No valid candidates found after applying min_probability_cutoff: {min_probability_cutoff}. Returning False.")
#         return False

#     top_candidate = sorted_valid_candidates[0]
#     top_candidate_name = top_candidate.get("name", "None")
#     top_probability = top_candidate.get("probability", 0)

#     # Check if the recognized candidate is indeed the top candidate
#     if recognized_candidate != top_candidate_name:
#         logger.info(f"Recognized candidate '{recognized_candidate}' is not the top candidate. Top candidate is '{top_candidate_name}'. Returning False.")
#         return False

#     # Check if the top candidate's probability exceeds the specified threshold
#     return top_probability >= threshold


# def is_concentrated_distribution(
#     quantity: str,
#     unit: str,
#     prototypedata: str,
#     quantity_with_probabilities: list,
#     unit_with_probabilities: list,
#     prototypedata_with_probabilities: list,
#     min_probability_cutoff: float,
#     threshold: float = 0.67,
#     **kwargs,
# ) -> dict:
#     """
#     Validates if the top candidate's probability is above a threshold after applying a minimal probability cutoff
#     for any of the provided candidate types (prototypedata, unit, quantity).

#     Args:
#         quantity (str): The recognized quantity class.
#         unit (str): The recognized unit class.
#         prototypedata (str): The recognized prototype data class.
#         quantity_with_probabilities (list): List of quantity candidates with aggregated probabilities.
#         unit_with_probabilities (list): List of unit candidates with aggregated probabilities.
#         prototypedata_with_probabilities (list): List of prototypedata candidates with aggregated probabilities.
#         min_probability_cutoff (float): The minimal probability for a candidate to be considered.
#                                         Candidates with average probability below this will be dropped.
#         threshold (float): The minimum probability for the top remaining candidate to be considered valid.
#                            Defaults to 0.67.

#     Returns:
#         dict: {"result": bool, "detail":{"quantity": bool, "unit": bool, "prototypeData": bool}}
#     """
#     logger.info("Checking for concentrated distribution of probabilities.")
#     logger.debug(f"Quantity probabilities: {quantity_with_probabilities}")
#     logger.debug(f"Unit probabilities: {unit_with_probabilities}")
#     logger.debug(f"PrototypeData probabilities: {prototypedata_with_probabilities}")

#     # Check if the top candidate for each type meets the probability threshold
#     # This now implicitly uses _get_sorted_valid_candidates within _is_top_candidate_above_threshold
#     quantity_meets_criteria = _is_top_candidate_above_threshold(quantity, quantity_with_probabilities, min_probability_cutoff, threshold)
#     unit_meets_criteria = _is_top_candidate_above_threshold(unit, unit_with_probabilities, min_probability_cutoff, threshold)
#     prototypeData_meets_criteria = _is_top_candidate_above_threshold(prototypedata, prototypedata_with_probabilities, min_probability_cutoff, threshold)

#     # Aggregate the results
#     result_detail = {"quantity": quantity_meets_criteria, "unit": unit_meets_criteria, "prototypeData": prototypeData_meets_criteria}
#     overall_result = quantity_meets_criteria and unit_meets_criteria and prototypeData_meets_criteria

#     toReturn = {
#         "result": overall_result,
#         "detail": result_detail,
#     }
#     logger.info(f"is_concentrated_distribution result: {toReturn}")
#     return toReturn


def judge_by_llm(
    user_query: str,
    recognition_result: dict,
    interpretation: str = "",
    rounds: int = 3,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
) -> float:
    """
    Judges the consistency of a recognition result with the original user query using an LLM.

    This function sends the user query, recognition result, and relevant terminologies
    to an LLM to determine if the recognition accurately reflects the user's intent.
    It aggregates probabilities from multiple rounds of LLM inference and checks if
    the average probability for a "match" scenario exceeds a specified threshold.

    Args:
        user_query (str): The original user query.
        recognition_result (dict): The dictionary containing the recognized classes
                                   (Quantity_class, Unit_class, PrototypeData_class).
        interpretation (str, optional): An interpretation of the user query, if available.
                                        Defaults to "".
        rounds (int, optional): The number of rounds to query the LLM for probabilities.
                                Defaults to 3.
        model (Optional[str], optional): The LLM model to use. Defaults to llm.DEFAULT_MODEL.
        user_config (Optional[dict], optional): User-specific configuration for the LLM.
                                                 Defaults to None.

    Returns:
        bool: True if the LLM judges the recognition result to match the user query
              (average probability >= threshold), False otherwise.
    """
    # Use default model if not specified
    if model is None:
        model = llm.DEFAULT_MODEL

    # Define the possible output options for the LLM's judgment
    output_options = ["Pass", "Fail"]
    option_explanations = {"Pass": "The recognition result matches the user query", "Fail": "The recognition result does not match the user query."}
    # Retrieve the appropriate prompt template for judgment from global configurations
    prompt_template = glb.prompt_template_collection["Generic"]["Judgement"]["judging"]

    # Prepare complementary knowledge string for the LLM prompt
    if complementary_knowledge is not None and "basic" in complementary_knowledge:
        complementary_knowledge_str = complementary_knowledge["basic"]
    else:
        complementary_knowledge_str = "No specific drilling terminology knowledge provided."

    # Extract recognized classes from the recognition result
    prototypeData = recognition_result["PrototypeData_class"]
    unit = recognition_result["Unit_class"]
    quantity = recognition_result["Quantity_class"]

    # Gather extra content/explanations for the recognized terminologies
    terminologies = {
        prototypeData: glb.prototypeData_fullList_extraContent.get(prototypeData, ""),
        unit: glb.unit_fullList_extraContent.get(unit, ""),
        quantity: glb.quantity_fullList_extraContent.get(quantity, ""),
    }

    # Prepare key-value pairs to populate the prompt template
    kvPairs = {
        "<user_query>": user_query,
        "<interpretation>": interpretation,
        "<recognition_result>": str({"Quantity_class": quantity, "Unit_class": unit, "PrototypeData_class": prototypeData}),
        "<terminologies>": terminologies,
        "<complementary_knowledge>": complementary_knowledge_str,
    }

    # Rank options by LLM probabilities to get the aggregated distribution
    probs_distribution_agg = pbs.rank_options_by_llm_probabilities(
        prompt_template=prompt_template,
        options=output_options,
        options_placeholder="<output_options>",
        option_explanations=option_explanations,
        option_explanations_placeholder="<option_explanations>",
        kvPairs=kvPairs,
        rounds=rounds,
    )

    average = 0
    # Iterate through the aggregated probabilities to find the average for the "match" option
    for pda in probs_distribution_agg:
        if output_options[0] in pda:
            average = pda[output_options[0]].get("average", 0)
            break

    logger.debug(f"Aggregated probability distribution from LLM: {probs_distribution_agg}")
    logger.debug(f"Average probability for 'The recognition result matches the user query': {average}")

    return average


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def method_voting(user_query: dict, ddbot_result: dict | None = None, top_k: int = 2) -> bool:
    # bm25_result, embedding_result and ddbot_result must contain keys_to_check.
    bm25_result = bm25.get_bm25_recognition_results(user_query, top_k)
    embedding_result = embedding.get_embedding_recognition_results(user_query, top_k)
    keys_to_check = ["Quantity_class", "Unit_class", "PrototypeData_class"]
    isConsistent = False
    # if ddbot_result is not None:
    #     isConsistent_1 = all(bm25_result.get(k) == ddbot_result.get(k) for k in keys_to_check)
    #     isConsistent_2 = all(embedding_result.get(k) == ddbot_result.get(k) for k in keys_to_check)
    # isConsistent = isConsistent_1 and isConsistent_2
    # A simple version
    isConsistent = all(bm25_result.get(k) == embedding_result.get(k) for k in keys_to_check)
    return isConsistent


def compute_id_score(
    matching_scores: dict,
    need_softmax: bool = True,
    non_id_labels: set | None = None,
    weights: list = [0.7, 0.1, 0.2],  # (MSP, Margin, 1-Entropy)
):
    # --- validations ---
    if len(weights) != 3:
        raise ValueError("weights must be a tuple of length 3 (w_msp, w_margin, w_sharp).")

    # normalize weights to sum=1 for geometric-mean semantics
    w = np.asarray(weights, dtype=float)
    w = w / (np.sum(w) + 1e-12)
    w_msp, w_margin, w_sharp = map(float, w)

    labels = list(matching_scores.keys())
    vals = np.array(list(matching_scores.values()), dtype=float)

    # 1) Probabilities
    if need_softmax:
        p = _softmax(vals)
    else:
        s = float(np.sum(vals))
        # if inputs already look like probs (≈1), keep; else renormalize defensively
        if not (0.999 <= s <= 1.001):
            p = vals / (s + 1e-12)
        else:
            p = vals

    # 2) Split ID / non-ID by non_id_labels (case-insensitive)
    if non_id_labels is None:
        non_id_labels_default = {
            "out-of-set",
            "uncertain",
            "OutOfSetUnit",
            "UncertainUnit",
            "OutOfSetQuantity",
            "UncertainQuantity",
            "UncertainPrototypeData",
            "OutOfSetPrototypeData",
            "oos",
        }
        nonid_set = {str(lbl).upper() for lbl in non_id_labels_default}
    else:
        nonid_set = {str(lbl).upper() for lbl in non_id_labels}

    upper = [str(lbl).upper() for lbl in labels]
    nonid_mask = np.array([u in nonid_set for u in upper], dtype=bool)
    id_mask = ~nonid_mask

    p_id = p[id_mask]
    # p_nonid = float(np.sum(p[nonid_mask])) if np.any(nonid_mask) else 0.0
    K = int(np.sum(id_mask))

    # 3) ID-confidence components
    msp = float(np.max(p_id)) if K > 0 else 0.0

    if K <= 1:
        margin = 1.0 if K == 1 else 0.0
    else:
        top2 = np.sort(p_id)[-2:]
        margin = float(top2[1] - top2[0])

    if K <= 1 or (p_id.sum() <= 1e-12):
        sharpness = 1.0
    else:
        pid_cond = p_id / (p_id.sum() + 1e-12)
        H = -np.sum(np.where(pid_cond > 0, pid_cond * np.log(pid_cond), 0.0))
        H_max = np.log(K + 1e-12)
        H_norm = float(H / (H_max + 1e-12))
        sharpness = 1.0 - H_norm

    # 4) Weighted geometric mean (with small floor)
    eps = 1e-9
    id_conf = np.exp(w_msp * np.log(max(msp, eps)) + w_margin * np.log(max(margin, eps)) + w_sharp * np.log(max(sharpness, eps)))
    id_conf = float(np.clip(id_conf, 0.0, 1.0))


def calculate_query_difficulty_score(isConsistent: bool) -> float:
    difficulty_score = 1.0 if isConsistent else 0.8
    return difficulty_score


def compute_nonid_score(
    matching_scores: dict,
    need_softmax: bool = True,
    non_id_labels: set | None = None,
    weights: list = [0.7, 0.1, 0.2],  # (MSP, Margin, 1-Entropy)
    gamma: float = 1.0,  # monotonic mapping strength: OOD_from_conf = 1 - id_conf**gamma
    blend_nonid: float = 0.8,  # fuse with explicit non-ID probability (sum over ood_labels)
    nonid_label_combine_mode: str = "max",  # one of {"blend","max","product"}
    query_difficulty_score: float = 1.0,
    llm_judge_score: float = 1.0,
) -> float:
    """
    Compute an ID confidence (internally) and an non-ID score (returned) from multi-class scores.

    Inputs
    ------
    matching_scores : dict
        Mapping from label -> score. Scores can be raw logits/similarities or already-normalized probs.
    need_softmax : bool
        If True, apply softmax to values to obtain probabilities. If False, values are treated as probs
        and will be renormalized defensively if they don't sum to 1.
    non_id_labels : iterable[str] | None
        Labels regarded as non-ID (OOD side). If None, defaults to {"out-of-set", "uncertain", "rejected","OOD", "ABSTAIN", "None"} (case-insensitive).
    weights : (w_msp, w_margin, w_sharp)
        Weights for combining MSP, Margin, and (1 - normalized entropy) via a weighted geometric mean.
        Must sum to 1 for a true geometric mean interpretation (not enforced).
    gamma : float
        Controls how aggressively low ID confidence converts to high nonid_score (larger => more conservative).
    blend_nonid : float in [0,1]
        Used in 'blend' mode: final nonid_score = (1 - id_conf**gamma)*(1-blend) + (p_nonid)*blend.
    nonid_label_combine_mode : {"blend","max","product"}
        Strategy to combine OOD-from-confidence with explicit non-ID probability:
        - "blend": convex combination (controlled by blend_nonid)
        - "max":   take max((1 - id_conf**gamma), p_nonid)  (more aggressive)
        - "product": (1 - id_conf**gamma) * p_nonid         (requires both signals to be high)

    Returns
    -------
    nonid_score : float in [0,1]
        Larger means more likely to be OOD.
    details : dict
        Diagnostic information including:
          - id_conf (the combined ID confidence)
          - msp, margin, sharpness
          - p_id_sum, p_nonid
          - weights, gamma, blend_nonid, combine_mode
    """

    # --- validations ---
    if len(weights) != 3:
        raise ValueError("weights must be a tuple of length 3 (w_msp, w_margin, w_sharp).")
    if not (0.0 <= blend_nonid <= 1.0):
        raise ValueError("blend_nonid must be in [0,1].")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if nonid_label_combine_mode not in {"blend", "max", "product"}:
        raise ValueError("nonid_label_combine_mode must be one of {'blend','max','product'}")

    # normalize weights to sum=1 for geometric-mean semantics
    w = np.asarray(weights, dtype=float)
    w = w / (np.sum(w) + 1e-12)
    w_msp, w_margin, w_sharp = map(float, w)

    labels = list(matching_scores.keys())
    vals = np.array(list(matching_scores.values()), dtype=float)

    # 1) Probabilities
    if need_softmax:
        p = _softmax(vals)
    else:
        s = float(np.sum(vals))
        # if inputs already look like probs (≈1), keep; else renormalize defensively
        if not (0.999 <= s <= 1.001):
            p = vals / (s + 1e-12)
        else:
            p = vals

    # 2) Split ID / non-ID by non_id_labels (case-insensitive)
    if non_id_labels is None:
        non_id_labels_default = {
            "out-of-set",
            "uncertain",
            "OutOfSetUnit",
            "UncertainUnit",
            "OutOfSetQuantity",
            "UncertainQuantity",
            "UncertainPrototypeData",
            "OutOfSetPrototypeData",
            "oos",
        }
        nonid_set = {str(lbl).upper() for lbl in non_id_labels_default}
    else:
        nonid_set = {str(lbl).upper() for lbl in non_id_labels}

    upper = [str(lbl).upper() for lbl in labels]
    nonid_mask = np.array([u in nonid_set for u in upper], dtype=bool)
    id_mask = ~nonid_mask

    p_id = p[id_mask]
    p_nonid = float(np.sum(p[nonid_mask])) if np.any(nonid_mask) else 0.0
    K = int(np.sum(id_mask))

    # 3) ID-confidence components
    msp = float(np.max(p_id)) if K > 0 else 0.0

    if K <= 1:
        margin = 1.0 if K == 1 else 0.0
    else:
        top2 = np.sort(p_id)[-2:]
        margin = float(top2[1] - top2[0])

    if K <= 1 or (p_id.sum() <= 1e-12):
        sharpness = 1.0
    else:
        pid_cond = p_id / (p_id.sum() + 1e-12)
        H = -np.sum(np.where(pid_cond > 0, pid_cond * np.log(pid_cond), 0.0))
        H_max = np.log(K + 1e-12)
        H_norm = float(H / (H_max + 1e-12))
        sharpness = 1.0 - H_norm

    # 4) Weighted geometric mean (with small floor)
    eps = 1e-9
    id_conf = np.exp(w_msp * np.log(max(msp, eps)) + w_margin * np.log(max(margin, eps)) + w_sharp * np.log(max(sharpness, eps)))
    id_conf = float(np.clip(id_conf, 0.0, 1.0))

    id_conf = id_conf * query_difficulty_score * llm_judge_score

    # 5) Map to OOD-like quantity
    nonid_from_conf = float(np.clip(1.0 - (id_conf**gamma), 0.0, 1.0))

    # 6) Combine with explicit non-ID probability
    if nonid_label_combine_mode == "blend":
        nonid_score = (1.0 - blend_nonid) * nonid_from_conf + blend_nonid * p_nonid
    elif nonid_label_combine_mode == "max":
        nonid_score = max(nonid_from_conf, p_nonid)
    else:  # "product"
        nonid_score = nonid_from_conf * p_nonid

    nonid_score = float(np.clip(nonid_score, 0.0, 1.0))

    return nonid_score


def simulate_nonid_score(recognition_result: str, non_id_labels: set | list | None = None):
    if non_id_labels is None:
        non_id_labels_default = {
            "out-of-set",
            "uncertain",
            "rejected",
            "OutOfSetUnit",
            "UncertainUnit",
            "OutOfSetQuantity",
            "UncertainQuantity",
            "UncertainPrototypeData",
            "OutOfSetPrototypeData",
            "oos",
        }
        nonid_set = {str(lbl).upper() for lbl in non_id_labels_default}
    else:
        nonid_set = {str(lbl).upper() for lbl in non_id_labels}

    nonid_score = 0.0
    if recognition_result.upper() in nonid_set:
        nonid_score = 0.9
    return nonid_score


# def need_human_intervention(user_query: str, recognition_result: dict) -> bool:
#     """
#     Determines if human intervention is required based on a series of validation rules.

#     This function orchestrates multiple checks to assess the quality and consistency
#     of a recognition result against the original user query and predefined knowledge.
#     The checks include:
#     1. Ontology Consistency: Ensures the recognized quantity, unit, and prototype data
#        are consistent with the knowledge base.
#     2. No "None" Recognition: Verifies that none of the primary recognized classes
#        (quantity, unit, prototype data) are explicitly the string "None".
#     3. Concentrated Probability Distribution: Checks if the top candidate for each
#        recognized class has a probability above a certain threshold, indicating
#        a clear and confident recognition.
#     4. LLM Judgment: Uses a Large Language Model to provide an overall judgment
#        on whether the recognition result accurately matches the user's query.

#     Args:
#         user_query (str): The original user query string.
#         recognition_result (dict): A dictionary containing the recognized classes
#                                    and their candidates, typically from a recognition process.
#                                    Example: {
#                                        "PrototypeData_class": "HookLoad",
#                                        "PrototypeData_class_candidates": ["HookLoad","None","WOB"],
#                                        "Quantity_class": "ForceQuantity",
#                                        "Quantity_class_candidates": ["ForceQuantity","None"],
#                                        "Unit_class": "ThousandKilogramForce",
#                                        "Unit_class_candidates": ["Decanewton","Kilodecanewton","Kilonewton","ThousandKilogramForce"]
#                                    }

#     Returns:
#         bool: True if human intervention is needed (i.e., any of the validation rules fail),
#               False otherwise (all rules pass).
#     """
#     # Access global configuration variables
#     global prompt_template_collection, complementary_knowledge, quantity_fullList_extraContent, unit_fullList_extraContent, prototypeData_fullList_extraContent

#     prompt_templates = prompt_template_collection["Generic"]

#     # Extract the primary recognized classes
#     quantity = recognition_result["Quantity_class"]
#     unit = recognition_result["Unit_class"]
#     prototypeData = recognition_result["PrototypeData_class"]

#     # Determine if recognition results include probabilities or just candidate names
#     recognition_result_has_probs = False
#     if isinstance(recognition_result["Quantity_class_candidates"], dict):
#         recognition_result_has_probs = True

#     # Extract candidates based on whether probabilities are included
#     if recognition_result_has_probs:
#         # If probabilities are present, candidates are keys of the nested dictionary
#         quantity_candidates = list(recognition_result["Quantity_class_candidates"].keys())
#         unit_candidates = list(recognition_result["Unit_class_candidates"].keys())
#         prototypeData_candidates = list(recognition_result["PrototypeData_class_candidates"].keys())
#     else:
#         # Otherwise, candidates are directly in the list
#         quantity_candidates = recognition_result["Quantity_class_candidates"]
#         unit_candidates = recognition_result["Unit_class_candidates"]
#         prototypeData_candidates = recognition_result["PrototypeData_class_candidates"]

#     # Helper function to rank candidates using LLM probabilities
#     def _rank_candidates(candidates: list, candidate_explanation: dict, rounds: int = 2):
#         return pbs.rank_options_tournament(
#             prompt_template=prompt_templates["Rank_for_validation"]["ranking"],
#             options=candidates,
#             options_placeholder="<candidates>",
#             option_explanations=candidate_explanation,
#             option_explanations_placeholder="<candidate_explanation>",
#             kvPairs={
#                 "<user_query>": user_query,
#                 "<interpretation>": "HKLO means Hookload",  # This interpretation seems hardcoded, consider making it dynamic if needed
#                 "<recognized_class>": str({"Quantity_class": quantity, "Unit_class": unit, "PrototypeData_class": prototypeData}),
#                 "<complementary_knowledge>": complementary_knowledge,
#             },
#             rounds=rounds,
#         )

#     # Ensure quantity_candidates is a list before proceeding
#     assert isinstance(quantity_candidates, list), "quantity_candidates is not a list"

#     # Rank candidates for each class to get probabilities
#     quantity_with_probabilities = _rank_candidates(
#         quantity_candidates, ut.supplement_candidates_extraContent(quantity_candidates, quantity_fullList_extraContent)
#     )
#     unit_with_probabilities = _rank_candidates(unit_candidates, ut.supplement_candidates_extraContent(unit_candidates, unit_fullList_extraContent))
#     prototypedata_with_probabilities = _rank_candidates(
#         prototypeData_candidates, ut.supplement_candidates_extraContent(prototypeData_candidates, prototypeData_fullList_extraContent)
#     )

#     # Apply validation rules
#     # Rule 1: Check for ontology consistency
#     pass_rule1 = validate_ontology_consistency(quantity, unit, prototypeData)

#     # Rule 2: Check if any recognized class is explicitly "None"
#     pass_rule2 = contains_no_none_recognition(quantity, unit, prototypeData)

#     # Rule 3: Check for concentrated probability distribution among candidates
#     check_result = is_concentrated_distribution(
#         quantity=quantity,
#         unit=unit,
#         prototypedata=prototypeData,
#         quantity_with_probabilities=quantity_with_probabilities,
#         unit_with_probabilities=unit_with_probabilities,
#         prototypedata_with_probabilities=prototypedata_with_probabilities,
#         min_probability_cutoff=0,  # Consider if this cutoff should be configurable
#         threshold=0.67,  # Consider if this threshold should be configurable
#     )
#     pass_rule3 = check_result.get("result", False)

#     # Rule 4: Get LLM's judgment on the overall recognition quality
#     pass_rule4 = judge_by_llm(user_query, recognition_result)

#     # Log the results of each rule for debugging and monitoring
#     logger.info(f"Rule 1 (Ontology Consistency) Pass?: {pass_rule1}")
#     logger.info(f"Rule 2 (No None Recognition) Pass?: {pass_rule2}")
#     logger.info(f"Rule 3 (Concentrated Distribution) Pass?: {pass_rule3}")
#     logger.info(f"Rule 4 (LLM Judgment) Pass?: {pass_rule4}")

#     # Human intervention is needed if any of the rules fail
#     return not (pass_rule1 and pass_rule2 and pass_rule3 and pass_rule4)


def _fill_with_probabilities(user_query, recognition_result):
    # Access global configuration variables
    global prompt_template_collection, complementary_knowledge, quantity_fullList_extraContent, unit_fullList_extraContent, prototypeData_fullList_extraContent

    prompt_templates = prompt_template_collection["Generic"]

    # Extract the primary recognized classes
    quantity = recognition_result["Quantity_class"]
    unit = recognition_result["Unit_class"]
    prototypeData = recognition_result["PrototypeData_class"]

    prompt_templates = prompt_template_collection["Generic"]

    quantity_candidates = list(recognition_result["Quantity_class_candidates"].keys())
    unit_candidates = list(recognition_result["Unit_class_candidates"].keys())
    prototypeData_candidates = list(recognition_result["PrototypeData_class_candidates"].keys())

    # Helper function to rank candidates using LLM probabilities
    def _rank_candidates(candidates: list, candidate_explanation: Union[str, dict], rounds: int = 2):
        return pbs.rank_options_by_llm_probabilities(
            prompt_template=prompt_templates["Rank_for_validation"]["ranking"],
            options=candidates,
            options_placeholder="<candidates>",
            option_explanations={},
            option_explanations_placeholder="N.A.",
            kvPairs={
                "<user_query>": user_query,
                "<recognized_class>": str({"Quantity_class": quantity, "Unit_class": unit, "PrototypeData_class": prototypeData}),
                "<complementary_knowledge>": complementary_knowledge,
                "<candidate_explanation>": str(candidate_explanation),
            },
            rounds=rounds,
        )

    # Ensure quantity_candidates is a list before proceeding
    assert isinstance(quantity_candidates, list), "quantity_candidates is not a list"

    # Rank candidates for each class to get probabilities
    quantity_with_probabilities = _rank_candidates(
        quantity_candidates, ut.supplement_candidates_extraContent(quantity_candidates, quantity_fullList_extraContent)
    )
    unit_with_probabilities = _rank_candidates(unit_candidates, ut.supplement_candidates_extraContent(unit_candidates, unit_fullList_extraContent))
    prototypedata_with_probabilities = _rank_candidates(
        prototypeData_candidates, ut.supplement_candidates_extraContent(prototypeData_candidates, prototypeData_fullList_extraContent)
    )
    return quantity_with_probabilities, unit_with_probabilities, prototypedata_with_probabilities


def assess_result_quality(user_query: str, recognition_result: dict, thresholds: dict) -> str:
    """
    Determines if human intervention is required based on a series of validation rules.

    This function orchestrates multiple checks to assess the quality and consistency
    of a recognition result against the original user query and predefined knowledge.
    The checks include:
    1. Ontology Consistency: Ensures the recognized quantity, unit, and prototype data
       are consistent with the knowledge base.
    2. No "None" Recognition: Verifies that none of the primary recognized classes
       (quantity, unit, prototype data) are explicitly the string "None".
    3. Concentrated Probability Distribution: Checks if the top candidate for each
       recognized class has a probability above a certain threshold, indicating
       a clear and confident recognition.
    4. LLM Judgment: Uses a Large Language Model to provide an overall judgment
       on whether the recognition result accurately matches the user's query.

    Args:
        user_query (str): The original user query string.
        recognition_result (dict): A dictionary containing the recognized classes
                                   and their candidates, typically from a recognition process.
                                   Example: {
                                       "PrototypeData_class": "HookLoad",
                                       "PrototypeData_class_candidates": ["HookLoad","None","WOB"],
                                       "Quantity_class": "ForceQuantity",
                                       "Quantity_class_candidates": ["ForceQuantity","None"],
                                       "Unit_class": "ThousandKilogramForce",
                                       "Unit_class_candidates": ["Decanewton","Kilodecanewton","Kilonewton","ThousandKilogramForce"]
                                   }

    Returns:
        str: Redo | Review | Accept
    """
    # Access global configuration variables
    global prompt_template_collection, complementary_knowledge, quantity_fullList_extraContent, unit_fullList_extraContent, prototypeData_fullList_extraContent

    # Extract the primary recognized classes
    quantity = recognition_result["Quantity_class"]
    unit = recognition_result["Unit_class"]
    prototypeData = recognition_result["PrototypeData_class"]
    if not validate_ontology_consistency(quantity, unit, prototypeData):
        return "Redo"

    #
    quantity_candidates = recognition_result["Quantity_class_candidates"]
    unit_candidates = recognition_result["Unit_class_candidates"]
    prototypeData_candidates = recognition_result["PrototypeData_class_candidates"]

    nonid_scores = {}
    nonid_scores["Quantity"] = compute_nonid_score(quantity_candidates, False)
    nonid_scores["Unit"] = compute_nonid_score(unit_candidates, False)
    nonid_scores["PrototypeData"] = compute_nonid_score(prototypeData_candidates, False)

    if thresholds is None:
        thresholds = {"Quantity": 0.7, "Unit": 0.7, "PrototypeData": 0.7}

    for task in ["Quantity", "Unit", "PrototypeData"]:
        if nonid_scores[task] >= thresholds[task]:
            return "Review"

    return "Accept"


def test():
    pass


if __name__ == "__main__":
    test()
