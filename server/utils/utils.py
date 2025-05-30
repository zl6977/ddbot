import logging
import random
import string
from collections import defaultdict
from difflib import get_close_matches
from typing import Optional

from . import chat_restapi as llm
from .configs import globals_config as glb

logger = logging.getLogger(__name__)


def filter_user_query(user_query: dict, selected_keys: list = ["Mnemonic", "Description"]) -> dict:
    """
    Reduce the distractions.
    """
    user_query_filtered = {}
    for k in user_query:
        if k in selected_keys:
            user_query_filtered[k] = user_query[k]
    return user_query_filtered


def remove_keys_from_dict(input_dict: dict, keys_to_remove: list) -> dict:
    filtered_dict = {}
    for k in input_dict:
        if k not in keys_to_remove:
            filtered_dict[k] = input_dict[k]
    return filtered_dict


def fix_project_id_list(raw_list: list):
    project_id_list = []
    for raw_id in raw_list:
        # normalized_path = path.replace("\\", "/")
        # project_id = os.path.basename(normalized_path)
        project_id = raw_id.replace("$", "_")
        project_id = project_id.replace("\\", "+")
        project_id = project_id.replace("/", "+")
        project_id = project_id.replace(" ", "_")
        project_id_list.append(project_id)
    return project_id_list


"""
user_query_minimal_keys = ["Namespace", "Mnemonic", "Description", "Unit", "DataType"]
A tyipical user query is:
user_query = {
    "Namespace": "",
    "Mnemonic": "",
    "Description": "",
    "Unit": "",
    "DataType": "",
    "Others": "",
}
"""

metadata_profile_dict = {
    "Volve open data": {
        "Namespace": "http://ddhub.demo/zzz/<project_id>",
        "selected_keys": [
            "mnemonic",
            "unit",
            "curveDescription",
            "dataSource",
            "typeLogData",
        ],
        "key_mapping": {
            "Mnemonic": "mnemonic",
            "Description": "curveDescription",
            "DataType": "typeLogData",
            "Unit": "unit",
        },
    },
    "mnemonic_rich_scraped": {
        "Namespace": "http://ddhub.demo/zzz/",
        "selected_keys": [
            "LongMnemonic",
            "ShortMnemonic",
            "shortDescription",
            "longDescription",
            "DataType",
            "DataLength",
            "MetricUnits",
            "FPSUnits",
        ],
        "key_mapping": {
            "Mnemonic": "LongMnemonic",
            "Description": "longDescription",
            "DataType": "DataType",
            "Unit": "MetricUnits",
        },
    },
}


def preprocess(s):
    s_after = s.replace("[", "").replace("]", "")
    s_after = s_after.replace(" ", "").replace("_", "").replace("-", "")
    s_after = s_after.lower()
    return s_after


def repair_terminology(targetStr: str, terminology_list: list) -> str:
    preprocessed_map = {preprocess(s): s for s in terminology_list}

    processed_target = preprocess(targetStr)

    if "Uncertain".lower() in processed_target or "OutOfSet".lower() in processed_target:
        return targetStr
    if processed_target in preprocessed_map:
        return preprocessed_map[processed_target]

    matches = get_close_matches(processed_target, preprocessed_map.keys(), n=1, cutoff=0.8)
    if matches:
        return preprocessed_map[matches[0]]
    else:
        print("no match found: " + targetStr)
        return "None"
    # raise ValueError("no match found: " + targetStr)


def repair_terminology_list(targetStr_list: list, terminology_list: list) -> list:
    repaired_list = []
    for targetStr in targetStr_list:
        repaired = repair_terminology(targetStr, terminology_list)
        repaired_list.append(repaired)
    return repaired_list


def assemble_prompt(prompt_template: str, kvPairs: dict) -> str:
    prompt = prompt_template
    for keyword in kvPairs.keys():
        if (kvPairs[keyword] is None) or (len(kvPairs[keyword]) == 0):
            continue
        prompt = prompt.replace(keyword, str(kvPairs[keyword]))
    return prompt


def run_rag_task_single(prompt: str, system_prompt: str = "", model: str = llm.DEFAULT_MODEL, user_config: Optional[dict] = None) -> dict:
    # prompt = assemble_prompt(prompt_template, kvPairs)
    response = llm.chat_with_llm(prompt, system_prompt=system_prompt, model=model, user_config=user_config)
    result = llm.result_extractor(response, model=model)
    logger.debug(f"prompt: {prompt}")
    logger.info(f"user_config: {user_config}")
    logger.debug(f"model: {result['model']}")
    logger.debug(f"content: {result['content']}")
    logger.debug(f"prompt_tokens: {result['prompt_tokens']}")
    logger.debug(f"completion_tokens: {result['completion_tokens']}")
    return result


def get_token_id(characters: str, model: str = llm.DEFAULT_MODEL) -> list:
    """
    Get the token IDs for a string of characters using the specified model's tokenizer.
    """
    import tiktoken

    encoding = tiktoken.encoding_for_model(model)
    token_ids = encoding.encode(characters)
    return token_ids


def supplement_candidates_extraContent(candidate_list: list, fullList_extraContent: dict, kickout_keys: Optional[list] = None) -> dict:
    candidates_extraContent = {}
    for c in candidate_list:
        if c == "none" or c == "None":
            continue
        extraContent = fullList_extraContent[c].copy()
        if kickout_keys is not None:
            for key in kickout_keys:
                if key in extraContent.keys():
                    extraContent.pop(key)
        candidates_extraContent.update({c: extraContent})
    return candidates_extraContent


# === Letter Mapping Functions ===


def generate_single_token_letters(num_needed: int, model: str = llm.DEFAULT_MODEL) -> list:
    """
    Generate single-token letter combinations for mapping candidates.

    Args:
        num_needed: Number of letter combinations needed
        model: LLM model to check tokenization against

    Returns:
        List of single-token letter combinations (A, B, ..., Z, AA, AB, ..., ZZ, etc.)
    """
    letters = []

    # Start with single letters A-Z
    for letter in string.ascii_uppercase:
        if check_is_single_token(letter, model):
            letters.append(letter)
        if len(letters) >= num_needed:
            return letters[:num_needed]

    # If we need more, try double letters AA, AB, AC, ..., ZZ
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            combo = first + second
            if check_is_single_token(combo, model):
                letters.append(combo)
            if len(letters) >= num_needed:
                return letters[:num_needed]

    # If we still need more, try triple letters AAA, AAB, etc.
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            for third in string.ascii_uppercase:
                combo = first + second + third
                if check_is_single_token(combo, model):
                    letters.append(combo)
                if len(letters) >= num_needed:
                    return letters[:num_needed]

    # If we still don't have enough, raise an error
    raise ValueError(f"Could not generate {num_needed} single-token letter combinations for model {model}")


def create_letter_mapping(candidate_list, model: str = llm.DEFAULT_MODEL, candidate_limit: int = 300):
    """Create a mapping between candidates and single-token letters/letter combinations."""
    num_candidates = len(candidate_list)

    try:
        candidate_letters = generate_single_token_letters(num_candidates, model)
    except ValueError as e:
        raise ValueError(f"Too many candidates ({num_candidates}) for single-token mapping: {e}")

    shuffled_candidates = shuffle_and_augment_candidates(candidate_list)
    if len(shuffled_candidates) > candidate_limit:
        logger.warning(f"Candidate list exceeded limit of {candidate_limit}. Truncating candidates to 300.")
        shuffled_candidates = shuffled_candidates[:candidate_limit]
    letter_map = {cand: candidate_letters[i] for i, cand in enumerate(shuffled_candidates)}
    reverse_map = {v: k for k, v in letter_map.items()}
    return letter_map, reverse_map


def shuffle_and_augment_candidates(candidate_list: list) -> list:
    """Shuffle the candidate list. Returns a new list."""
    candidates = list(candidate_list)
    random.shuffle(candidates)
    return candidates


# === Ranking Functions (adapted from rank_candidates.py) ===


def check_is_single_token(text: str, model: str = llm.DEFAULT_MODEL) -> bool:
    """Check if a string is encoded as a single token by the model's tokenizer."""
    try:
        token_ids = get_token_id(text, model)
        return len(token_ids) == 1
    except Exception:
        # If tokenization fails, assume it's not a single token
        return False


def supplement_quantity_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> dict:
    """
    Supplement quantity candidates with extra content.

    Args:
        original_candidates: Original candidate list
        supplementary_candidates: Additional candidates to include
        fullList_extraContent: Full knowledge base content dict

    Returns:
        Dict mapping candidate names to their extra content
    """
    candidates = original_candidates + supplementary_candidates
    # Which unit should be handled by subsequent ontology pruning
    candidates_extraContent = supplement_candidates_extraContent(candidates, fullList_extraContent)
    return candidates_extraContent


def supplement_unit_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> dict:
    """
    Supplement unit candidates with extra content.

    Args:
        original_candidates: Original candidate list
        supplementary_candidates: Additional candidates to include
        fullList_extraContent: Full knowledge base content dict

    Returns:
        Dict mapping candidate names to their extra content
    """
    candidates = original_candidates + supplementary_candidates
    # Unit is given before, this should be just the `extraContent` of the unit
    candidates_extraContent = supplement_candidates_extraContent(candidates, fullList_extraContent)
    return candidates_extraContent


def supplement_prototypeData_candidates(
    original_candidates: list,
    supplementary_candidates: list,
    fullList_extraContent: dict,
) -> dict:
    """
    Supplement prototypeData candidates with extra content.

    Args:
        original_candidates: Original candidate list
        supplementary_candidates: Additional candidates to include
        fullList_extraContent: Full knowledge base content dict

    Returns:
        Dict mapping candidate names to their extra content
    """
    candidates = original_candidates + supplementary_candidates
    # IsOfMeasurableQuantity should not be sent to LLM in any cases. PrototypeData is given before, this should be just the `extraContent` of the prototypeData
    candidates_extraContent = supplement_candidates_extraContent(candidates, fullList_extraContent, ["ddhub:IsOfMeasurableQuantity"])
    return candidates_extraContent


def retrieve_unit_relatedTo_Quantiy(quantity_candidates: list) -> dict:
    unit_list_related = defaultdict(list)
    for qc in quantity_candidates:
        unit_list_tmp = glb.quantity_fullList_extraContent.get(qc, {}).get("zzz:QuantityHasUnit", [])
        for unit in unit_list_tmp:
            unit_list_related[unit].append(qc)
    return unit_list_related


def retrieve_prototypeData_relatedTo_Quantiy(quantity_candidates: list) -> dict:
    prototypeData_list_related = defaultdict(list)
    for qc in quantity_candidates:
        prototypeData_list_tmp = glb.quantity_fullList_extraContent.get(qc, {}).get("zzz:PrototypeData", [])
        for prototypeData in prototypeData_list_tmp:
            prototypeData_list_related[prototypeData].append(qc)
    return prototypeData_list_related


def enrich_classes_with_extra_content(recognized_classes: dict) -> dict:
    """
    Enrich recognized classes with extra content from knowledge base.

    Args:
        recognized_classes: Dict containing recognized class names

    Returns:
        Dict with enriched classes containing extra content
    """

    enriched_classes = recognized_classes.copy()

    # Enrich Quantity class
    if "Quantity_class" in recognized_classes:
        quantity_class = recognized_classes["Quantity_class"]
        if quantity_class and quantity_class in glb.quantity_fullList_extraContent:
            streamlined_extraContent = {k: v for k, v in glb.quantity_fullList_extraContent[quantity_class].items() if k != "ddhub:Quantity"}
            streamlined_extraContent["Value Explanation"] = streamlined_extraContent.pop("rdfs:comment", None)
            streamlined_extraContent["Associated Units"] = streamlined_extraContent.pop("zzz:QuantityHasUnit", [])
            streamlined_extraContent["Associated PrototypeData"] = streamlined_extraContent.pop("zzz:PrototypeData", [])
            enriched_classes["Quantity_class"] = {quantity_class: streamlined_extraContent}
            logger.debug(f"Enriched Quantity_class '{quantity_class}' with extra content")

    # Enrich Unit class
    if "Unit_class" in recognized_classes:
        unit_class = recognized_classes["Unit_class"]
        if unit_class and unit_class in glb.unit_fullList_extraContent:
            streamlined_extraContent = {k: v for k, v in glb.unit_fullList_extraContent[unit_class].items() if k != "ddhub:Unit"}
            streamlined_extraContent["Value Explanation"] = streamlined_extraContent.pop("rdfs:comment", None)
            streamlined_extraContent["Associated Quantities"] = streamlined_extraContent.pop("ddhub:IsUnitForQuantity", [])
            streamlined_extraContent["Common Mnemonics"] = streamlined_extraContent.pop("zzz:commonMnemonics", [])
            enriched_classes["Unit_class"] = {unit_class: streamlined_extraContent}
            logger.debug(f"Enriched Unit_class '{unit_class}' with extra content")

    # Enrich PrototypeData class
    if "PrototypeData_class" in recognized_classes:
        prototype_class = recognized_classes["PrototypeData_class"]
        if prototype_class and prototype_class in glb.prototypeData_fullList_extraContent:
            streamlined_extraContent = {
                k: v
                for k, v in glb.prototypeData_fullList_extraContent[prototype_class].items()
                if k != "ddhub:PrototypeData" and k != "ddhub:IsOfMeasurableQuantity"
            }
            streamlined_extraContent["Value Explanation"] = streamlined_extraContent.pop("rdfs:comment", None)
            streamlined_extraContent["Associated Quantities"] = streamlined_extraContent.pop("ddhub:IsOfBaseQuantity", [])
            streamlined_extraContent["Common Mnemonics"] = streamlined_extraContent.pop("zzz:commonMnemonics", [])
            enriched_classes["PrototypeData_class"] = {prototype_class: streamlined_extraContent}
            logger.debug(f"Enriched PrototypeData_class '{prototype_class}' with extra content")

    logger.debug(f"Classes enriched with extra content: {list(enriched_classes.keys())}")
    return enriched_classes
