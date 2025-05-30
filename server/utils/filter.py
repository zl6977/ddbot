import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# Example rules
def pass_rule_1(metadata: dict) -> bool:
    """
    Check if it is a composite measurement.
    """
    mnemonic = metadata.get("Mnemonic", "")
    pattern = r"[+\-*/]"
    return re.search(pattern, mnemonic) is None


def pass_rule_2(metadata: dict) -> bool:
    """
    Check if it is a numeric measurement.
    """
    dataType = metadata.get("DataType", "")
    pass_rule = True
    rejected_types = ["string", "bool", "boolean"]
    if dataType.lower() in rejected_types:
        pass_rule = False
    return pass_rule


def filter_metadata(metadata: Dict[str, Any], rules: List[Callable[[Dict[str, Any]], bool]] = [pass_rule_1, pass_rule_2]) -> Tuple:
    """
    Filter dictionary entries based on multiple rules.

    :param metadata: The input dictionary.
    :param rules: A list of functions, each taking (key, value) and returning True if the item should be kept.
    :return: A new dictionary with items that pass all rules.

    A metadata example is:
    data = {
        "SPPA": {"Mnemonic": "SPPA", "Description": "Standpipe pressure.", "Unit": "Pa", "Namespace": "witsml_demo_202508"},
        "ROP": {"Mnemonic": "ROP", "Description": "ROP", "Unit": "m/s", "Namespace": "witsml_demo_202508"},
    }
    """

    def passes_all_rules(value: Dict[str, Any]) -> bool:
        return all(rule(value) for rule in rules)

    rejected, toProcess = {}, {}
    for k, v in metadata.items():
        if passes_all_rules(v):
            toProcess.update({k: v})
        else:
            rejected.update({k: v})

    return toProcess, rejected


def test():
    # Apply multiple rules
    rules = [pass_rule_1, pass_rule_2]

    # Example metadata
    # metadata = {
    #     "SPPA": {"Mnemonic": "SPPA", "Description": "Standpipe pressure.", "Unit": "Pa", "Namespace": "witsml_demo_202508"},
    #     "ROP": {"Mnemonic": "ROP", "Description": "ROP", "Unit": "m/s", "Namespace": "witsml_demo_202508"},
    # }

    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent.parent
    query_files = [
        project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+1+1+1+00001.json",
        project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+1+1+00001.json",
        project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+2+1+00001.json",
    ]

    query_objs = []
    for i in range(len(query_files)):
        with open(query_files[i], "r", encoding="utf-8") as f:
            data = json.load(f)
            query_tmp = {}
            for j in range(len(data)):
                mnemonic = data[j]["Mnemonic"]
                query_tmp.update({mnemonic: data[j]})
            query_objs.append(query_tmp)

        filtered, rejected = filter_metadata(query_objs[i], rules)
        # print("filtered", filtered)
        print("rejected", rejected)
        print("count", len(rejected))


if __name__ == "__main__":
    test()
