import json
import logging
import os
from collections import defaultdict

# import preprocessor as pps
from typing import List, Optional, Union

import yaml

from .. import chat_restapi as llm
from .. import sparql_connector as sc
from .. import utils as ut
from ..configs import log_config

logger = logging.getLogger(__name__)



def remove_candidates_extraContent_list(candidates_extraContent:  Optional[list]) -> list:
    candidates = []
    if candidates_extraContent is None or not candidates_extraContent:
        return candidates
    sample = candidates_extraContent[0]
    target, containExtraContent = "", False
    if "ddhub:Quantity" in sample:
        target = "ddhub:Quantity"
        containExtraContent = True
    elif "ddhub:Unit" in sample:
        target = "ddhub:Unit"
        containExtraContent = True
    elif "ddhub:PrototypeData" in sample:
        target = "ddhub:PrototypeData"
        containExtraContent = True
    else:
        # candidates_extraContent does not contain extra content
        return candidates_extraContent

    for c in candidates_extraContent:
        if containExtraContent:
            candidates.append(c[target])
    return candidates

def remove_candidates_extraContent_dict(candidates_extraContent: Optional[dict]) -> dict:
    candidates = defaultdict(list)
    if candidates_extraContent is None or not candidates_extraContent:
        return candidates
    sample = next(iter(candidates_extraContent.values()))
    target, containExtraContent = "", False
    if "ddhub:Quantity" in sample:
        target = "ddhub:Quantity"
        containExtraContent = True
    elif "ddhub:Unit" in sample:
        target = "ddhub:Unit"
        containExtraContent = True
    elif "ddhub:PrototypeData" in sample:
        target = "ddhub:PrototypeData"
        containExtraContent = True
    else:
        # candidates_extraContent does not contain extra content
        return candidates_extraContent

    for c, extra_content in candidates_extraContent.items():
        if containExtraContent:
            candidates[c].append(extra_content[target])
    return candidates

def remove_candidates_extraContent(candidates_extraContent: Optional[dict] | Optional[list]) -> dict | list:
    if candidates_extraContent is None or not candidates_extraContent:
        return []
    if isinstance(candidates_extraContent, list):
        return remove_candidates_extraContent_list(candidates_extraContent)
    elif isinstance(candidates_extraContent, dict):
        return remove_candidates_extraContent_dict(candidates_extraContent)
    else:
        raise ValueError("candidates_extraContent should be a list or a dict.")
    


def narrow_selection_range(
    task_type: str,
    prompt_templates: dict,
    metadata: Optional[Union[dict, str]] = None,
    selection_range: Optional[list] | Optional[dict] = None,
    interpretation: Optional[str] = None,
    complementary_knowledge: Optional[dict] = None,
    other_pairs: Optional[dict] = None,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
    prompt_template_entry = prompt_templates[task_type]["stages"]
    prompt_template_analysis = prompt_template_entry["analysis"]
    prompt_template_extraction = prompt_template_entry["extraction"]

    kvPairs = prompt_templates[task_type].get("placeholders", [])
    kvPairs = {k: "" for k in kvPairs}

    # complementary_knowledge = prompt_templates["complementary_knowledge"]["basic"]
    if complementary_knowledge is not None:
        complementary_knowledge_str = complementary_knowledge["basic"]
    else:
        complementary_knowledge_str = ""

    if metadata is None:
        logger.error("user_query is not provided!")
        raise ValueError("user_query is not provided!")
    user_query_str = metadata if isinstance(metadata, str) else str(metadata)

    kvPairs.update({
        "<user_query>": user_query_str,
        "<complementary_knowledge>": complementary_knowledge_str,
    })
    if interpretation is not None:
        kvPairs.update({"<interpretation>": interpretation})
    if selection_range is not None:
        if isinstance(selection_range, list):
            kvPairs.update({"<selection_range>": "\n".join(map(str, selection_range))})
        else:
            kvPairs.update({"<selection_range>": json.dumps(selection_range, ensure_ascii=False, indent=2)})
    if other_pairs is not None:
        kvPairs.update(other_pairs)

    prompt = ut.assemble_prompt(prompt_template_analysis, kvPairs)
    result = ut.run_rag_task_single(prompt, model=model)
    last_answer = result["content"] if result and "content" in result else ""
    kvPairs.update({"<last_answer>": last_answer})
    logger.info(f"{task_type} analysis prompt:\n{prompt}")
    # logger.info(f"{task_type} analysis result:\n{last_answer}")

    if selection_range:
        candidated_without_extraContent = remove_candidates_extraContent(selection_range)
        if isinstance(selection_range, list):
            kvPairs.update({"<selection_range>": "\n".join(map(str, candidated_without_extraContent))})
        else:
            kvPairs.update({"<selection_range>": json.dumps(candidated_without_extraContent, ensure_ascii=False, indent=2)})
    prompt = ut.assemble_prompt(prompt_template_extraction, kvPairs)
    result = ut.run_rag_task_single(prompt, model=model)

    content_to_split = result["content"] if result and "content" in result and result["content"] is not None else ""
    result_list = content_to_split.split(",")
    result_content = [s.strip() for s in result_list]
    logger.info(f"{task_type} extraction prompt:\n{prompt}")
    logger.info(f"{task_type} extraction result:\n{result_content}")
    return result_content, prompt


def generate_fullList_extraContent(KB_location: str) -> tuple:
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(KB_location)
    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(KB_location)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(KB_location)
    return (
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
    )


def run_task_single(
    user_query: dict,
    complementary_knowledge: dict,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    unit_fullList_extraContent: dict,
    prototypeData_fullList_extraContent: dict,
    interpretation: Optional[str] = None,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
    **kwargs,
):
    """
    Preparation: the full list with extra content as cache: quantity, prototypeData, unit.

    1. Interpret the metadata.
    2. Recognize the quantity.
        1. Retrieve all the quantities from KB.
        2. Use LLM to preselect top X candidates from the full list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize quantity.
    3. Recognize the unit.
        1. Retrieve the units related to top X quantity candidates from KB.
        2. Use LLM to preselect top X candidates from the related list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize unit.
    4. Recognize the prototypeData.
        1. Retrieve the prototypeData related to top X quantity candidates from KB.
        2. Use LLM to preselect top X candidates from the related list.
        3. Adjust the candidate list and supplement extra content.
        4. Use LLM to finally recognize prototypeData.
    5. Retrieve the MeasurableQuantity class from KB.
    """
    # quantity_fullList = list(quantity_fullList_extraContent.keys())
    # unit_fullList = list(unit_fullList_extraContent.keys())
    # prototypeData_fullList = list(prototypeData_fullList_extraContent.keys())

    # 1. Interpret the metadata.
    if interpretation is None:
        interpretation, prompt_interpret_mnemonic = interpret_mnemonic(str(user_query), complementary_knowledge, prompt_templates, models_high_low[1])
    # print("--------\n", "interpretation: ", interpretation, "\n")
    # logger.info(f"interpretation: {interpretation}")

    # Assert that interpretation is a string. This helps type checkers.
    assert interpretation is not None, "Interpretation should not be None at this point."

    # 2, 3, 4, 5. Recognize the quantity, unit, prototypeData.
    [recognized_class, candidates, prompts] = recoginize_metadata(
        user_query,
        interpretation,
        complementary_knowledge,
        prompt_templates,
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
        models_high_low,
    )

    prompts.update({"prompt_interpret_mnemonic": prompt_interpret_mnemonic})

    toReturn = [interpretation, recognized_class, candidates, prompts]
    return toReturn


def recoginize_metadata(
    user_query: dict,
    interpretation: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    quantity_fullList_extraContent: dict,
    unit_fullList_extraContent: dict,
    prototypeData_fullList_extraContent: dict,
    models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL],
    number_of_candidates: Optional[dict] = None,
    **kwarg,
) -> tuple:
    # Results to return
    # recognized_class, candidates, prompts = {}, {}, {}

    # Set default candidate numbers if not provided
    if number_of_candidates is None:
        number_of_candidates = {
            "Quantity_class": 5,
            "Unit_class": 10,
            "PrototypeData_class": 5,
        }

    quantity_fullList = list(quantity_fullList_extraContent.keys())
    # unit_fullList = list(unit_fullList_extraContent.keys())
    # prototypeData_fullList = list(prototypeData_fullList_extraContent.keys())

    # 2.1. Retrieve all the quantities from KB.
    # 2.2. Use LLM to preselect top X candidates from the full list.
    quantity_candidates, prompt_preselect_quantity = preselect_quantity(
        user_query, quantity_fullList, prompt_templates, interpretation, complementary_knowledge, models_high_low[1],
        number_of_candidates.get("Quantity_class", 5)
    )
    # print("quantity_candidates: ", quantity_candidates)
    logger.info(f"quantity_candidates: {quantity_candidates}")

    # 2.3. Adjust the candidate list and supplement extra content.
    if not quantity_candidates:
        quantity_candidates = list(quantity_fullList_extraContent.keys())
    quantity_candidates_extraContent = ut.supplement_quantity_candidates(quantity_candidates, [], quantity_fullList_extraContent)
    # 2.4. Use LLM to finally recognize quantity.
    quantity_class, prompt_recognize_quantity = recognize_quantity(
        user_query, quantity_candidates_extraContent, prompt_templates, interpretation, complementary_knowledge, models_high_low[1]
    )
    # print("quantity_class: ", quantity_class, "\n---")
    logger.info(f"quantity_class: {quantity_class}")

    # 3.1. Retrieve the units related to top X quantity candidates from KB.
    quantity_candidates = list(quantity_candidates_extraContent.keys())
    unit_candidates_related = ut.retrieve_unit_relatedTo_Quantiy(quantity_candidates)
    # print("unit_candidates_related", unit_candidates_related)
    if not unit_candidates_related:
        unit_candidates_related = unit_fullList_extraContent

    # 3.2. Use LLM to preselect top X candidates from the related list.
    unit_candidates_preselected, prompt_preselect_unit = preselect_unit(
        user_query, unit_candidates_related, prompt_templates, interpretation, complementary_knowledge, models_high_low[1],
        number_of_candidates.get("Unit_class", 10)
    )

    # 3.3. Adjust the candidate list and supplement extra content.
    if not unit_candidates_preselected:
        unit_candidates_preselected = list(unit_fullList_extraContent.keys())
    unit_candidates_extraContent = ut.supplement_unit_candidates(unit_candidates_preselected, [], unit_fullList_extraContent)
    # print("unit_candidates: ", list(unit_candidates_extraContent.keys()))
    logger.info(f"unit_candidates: {list(unit_candidates_extraContent.keys())}")

    # 3.4. Use LLM to finally recognize unit.
    unit_class, prompt_recognize_unit = recognize_unit(
        user_query, unit_candidates_extraContent, prompt_templates, interpretation, complementary_knowledge, models_high_low[1]
    )
    # print("unit_class: ", unit_class, "\n---")
    logger.info(f"unit_class: {unit_class}")

    # 4.1. Retrieve the prototypeData related to top X quantity candidates from KB.
    prototypeData_candidates_related = ut.retrieve_prototypeData_relatedTo_Quantiy(quantity_candidates)
    if not prototypeData_candidates_related:
        prototypeData_candidates_related = prototypeData_fullList_extraContent
    # 4.2. Use LLM to preselect top X candidates from the related list.
    prototypeData_candidates_preselected, prompt_preselect_prototypeData = preselect_prototypeData(
        user_query, prototypeData_candidates_related, prompt_templates, interpretation, complementary_knowledge, models_high_low[0],
        number_of_candidates.get("PrototypeData_class", 5)
    )
    if not prototypeData_candidates_preselected:
        prototypeData_candidates_preselected = list(prototypeData_fullList_extraContent.keys())

    # 4.3. Adjust the candidate list and supplement extra content.
    prototypeData_candidates_extraContent = ut.supplement_prototypeData_candidates(
        prototypeData_candidates_preselected,
        [],
        prototypeData_fullList_extraContent,
    )
    # print("prototypeData_candidates: ", list(prototypeData_candidates_extraContent.keys()))
    logger.info(f"prototypeData_candidates: {list(prototypeData_candidates_extraContent.keys())}")

    # 4.4. Use LLM to finally recognize prototypeData.
    prototypeData_class, prompt_recognize_prototypeData = recognize_prototypeData(
        user_query, prototypeData_candidates_extraContent, prompt_templates, interpretation, complementary_knowledge, models_high_low[0]
    )
    # print("prototypeData_class: ", prototypeData_class, "\n---")
    logger.info(f"prototypeData_class: {prototypeData_class}")

    # 5. Retrieve the MeasurableQuantity class from KB.
    if prototypeData_class != "None":
        MQuantity_class = prototypeData_fullList_extraContent[prototypeData_class]["ddhub:IsOfMeasurableQuantity"]
    else:
        MQuantity_class = "None"

    # package the results to return
    recognized_class = {
        "Quantity_class": quantity_class,
        "Unit_class": unit_class,
        "PrototypeData_class": prototypeData_class,
        "MeasurableQuantity_class": MQuantity_class,
    }

    # Candidates and their probabilities, by default, all probabilities are 0.0.
    candidates = {
        "Quantity_candidates": {candidate: 0.0 for candidate in quantity_candidates},
        "Unit_candidates": {candidate: 0.0 for candidate in unit_candidates_extraContent.keys()},
        "PrototypeData_candidates": {candidate: 0.0 for candidate in prototypeData_candidates_extraContent.keys()},
    }
    # Set the recognized candidates' probabilities to 1.0.
    candidates["Quantity_candidates"][quantity_class] = 1.0
    candidates["Unit_candidates"][unit_class] = 1.0
    candidates["PrototypeData_candidates"][prototypeData_class] = 1.0

    prompts = {
        "prompt_preselect_quantity": prompt_preselect_quantity,
        "prompt_recognize_quantity": prompt_recognize_quantity,
        "prompt_preselect_unit": prompt_preselect_unit,
        "prompt_recognize_unit": prompt_recognize_unit,
        "prompt_preselect_prototypeData": prompt_preselect_prototypeData,
        "prompt_recognize_prototypeData": prompt_recognize_prototypeData,
    }

    toReturn = (recognized_class, candidates, prompts)
    return toReturn


def interpret_mnemonic(
    user_query: str,
    complementary_knowledge: dict,
    prompt_templates: dict,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
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

    kvPairs.update({
        "<user_query>": user_query_str,
        "<complementary_knowledge>": complementary_knowledge_str,
    })
    prompt = ut.assemble_prompt(prompt_template_analysis, kvPairs)
    result = ut.run_rag_task_single(prompt, model=model)

    logger.info(f"Interpret_mnemonic analysis prompt:\n{prompt}")

    kvPairs.update({"<last_answer>": result["content"]})
    prompt = ut.assemble_prompt(prompt_template_extraction, kvPairs)
    result = ut.run_rag_task_single(prompt, model=model)

    interpretation = str(result["content"])

    logger.info(f"Interpret_mnemonic extraction prompt:\n{prompt}")
    logger.info(f"Interpret_mnemonic extraction result:\n{interpretation}")

    # interpretation, prompt = narrow_selection_range("Interpret_mnemonic", prompt_templates, metadata)
    return interpretation, prompt


def preselect_quantity(
    metadata: dict,
    quantity_fullList: list,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
    number_of_candidates: int = 5,
) -> tuple:
    task_type = "Preselect_quantity"
    other_pairs = {"<num_quantity_candidates>": str(number_of_candidates)}
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        quantity_fullList,
        interpretation,
        complementary_knowledge,
        other_pairs=other_pairs,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    preselected_quantity_list = ut.repair_terminology_list(result_content, quantity_fullList)
    if len(preselected_quantity_list) == 1 and "UncertainQuantity" in preselected_quantity_list:
        preselected_quantity_list = quantity_fullList
    return preselected_quantity_list, prompt


def recognize_quantity(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
    task_type = "Recognize_quantity"
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        list(candidates_extraContent.values()),
        interpretation,
        complementary_knowledge,
        model=model,
    )
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    repaired_term = ut.repair_terminology_list(result_content, list(candidates_extraContent.keys()))
    return repaired_term[0], prompt


def preselect_unit(
    metadata: dict,
    unit_fullList: dict,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
    number_of_candidates: int = 10,
) -> tuple:
    task_type = "Preselect_unit"
    other_pairs = {"<num_unit_candidates>": str(number_of_candidates)}
    unit_str_list = [f"Unit '{k}' Related To: '{v}'" for k, v in unit_fullList.items()]
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        unit_str_list,
        interpretation,
        complementary_knowledge,
        other_pairs=other_pairs,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    # repair the terminology list
    preselected_quantity_list = ut.repair_terminology_list(result_content, list(unit_fullList.keys()))
    return preselected_quantity_list, prompt


def recognize_unit(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
    task_type = "Recognize_unit"
    selection_range = list(candidates_extraContent.values())
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        selection_range,
        interpretation,
        complementary_knowledge,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    repaired_term = ut.repair_terminology_list(result_content, list(candidates_extraContent.keys()))
    return repaired_term[0], prompt


def preselect_prototypeData(
    metadata: dict,
    prototypeData_fullList: dict,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
    number_of_candidates: int = 5,
) -> tuple:
    task_type = "Preselect_prototypeData"
    other_pairs = {"<num_prototypedata_candidates>": str(number_of_candidates)}
    prototypeData_str_list = [f"Prototype Data '{k}' Related To: '{v}'" for k, v in prototypeData_fullList.items()]
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        prototypeData_str_list,
        interpretation,
        complementary_knowledge,
        other_pairs=other_pairs,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    # repair the terminology list
    preselect_prototypeData_list = ut.repair_terminology_list(result_content, list(prototypeData_fullList.keys()))
    return preselect_prototypeData_list, prompt

def recognize_prototypeData(
    metadata: dict,
    candidates_extraContent: dict,
    prompt_templates: dict,
    interpretation: str,
    complementary_knowledge: dict,
    model: str = llm.DEFAULT_MODEL,
) -> tuple:
    task_type = "Recognize_prototypeData"
    selection_range = list(candidates_extraContent.values())
    # for c_ex in list(candidates_extraContent.values()):
    #     extra_content = utils.remove_keys_from_dict(c_ex, ["ddhub:IsOfMeasurableQuantity"])
    #     selection_range.append(extra_content)
    result_content, prompt = narrow_selection_range(
        task_type,
        prompt_templates,
        metadata,
        selection_range,
        interpretation,
        complementary_knowledge,
        model=model,
    )
    # print("prompt: ", prompt)
    # print("result_content: ", result_content, "\n")
    # logger.info(f"{task_type} prompt:\n{prompt}")
    # logger.info(f"{task_type} result: {result_content}")
    repaired_term = ut.repair_terminology_list(result_content, list(candidates_extraContent.keys()))
    return repaired_term[0], prompt


def test():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    log_config.configure_logger(currentFolder + r"/../tmp/test.log")

    user_query = {
        "Mnemonic": "HKLO",
        "Description": "zzz:undefined",
        "DataType": "double",
        "Unit": "kkgf",
    }

    prompt_templates_path = currentFolder + "/prompt_resources/templates_pc_cot.yaml"
    with open(prompt_templates_path, "r") as json_file:
        prompt_templates = yaml.safe_load(json_file)

    complementary_knowledge_path = currentFolder + "/../../data_store/complementary_knowledge.yaml"
    with open(complementary_knowledge_path, "r") as json_file:
        complementary_knowledge = yaml.safe_load(json_file)

    logger.info("Start retrieving knowledge.")
    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(sc.KB_ttl_path)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(sc.KB_ttl_path)
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(sc.KB_ttl_path)
    logger.info("Finish retrieving knowledge.")

    run_task_single(
        user_query,
        complementary_knowledge,
        prompt_templates,
        quantity_fullList_extraContent,
        unit_fullList_extraContent,
        prototypeData_fullList_extraContent,
    )


if __name__ == "__main__":
    test()
