import json
import os
import xml.etree.ElementTree as ET
from typing import Callable

# import yaml

from utils import metadata_profile_dict

metadata_profile_mnemonic_rich = metadata_profile_dict["mnemonic_rich_scraped"]
metadata_profile_onsite = metadata_profile_dict["Volve open data"]


def read_from_xml_onsite(xml_file_path: str):
    # Read the XML content from the file
    with open(xml_file_path, "r", encoding="utf-8") as file:
        xml_content = file.read()
    # Parse the XML
    root = ET.fromstring(xml_content)

    namespace = {"ns": "http://www.witsml.org/schemas/1series"}
    preprocessable_metadata_dict = {}
    for log_curve in root.iterfind(".//ns:logCurveInfo", namespaces=namespace):
        uid = log_curve.get("uid")
        preprocessable_metadata_dict[uid] = {child.tag.split("}")[-1]: child.text for child in log_curve}
    return preprocessable_metadata_dict


def read_from_json_default(json_file_path: str):
    # Read the XML content from the file
    with open(json_file_path, "r") as json_file:
        preprocessable_metadata_dict = json.load(json_file)
    return preprocessable_metadata_dict


def preprocess_metadata(
    preprocessable_metadata: dict, metadata_profile: dict = metadata_profile_mnemonic_rich
) -> dict:
    selected_keys = metadata_profile["selected_keys"]
    key_mapping = metadata_profile["key_mapping"]
    keys_user_query = list(key_mapping.keys())
    keys_mapped = list(key_mapping.values())

    preprocessed_metadata = {}
    for key in keys_user_query:
        key_mapped = key_mapping[key]
        if key_mapped in preprocessable_metadata:
            preprocessed_metadata[key] = preprocessable_metadata[key_mapped]
        else:
            preprocessed_metadata[key] = "zzz:undefined"

    for key, value in preprocessable_metadata.items():
        if (key in selected_keys) and ((key not in keys_mapped)):
            preprocessed_metadata[key] = value
    return preprocessed_metadata


# def preprocess_metadata_batch_0(raw_metadata_batch: dict, metadata_profile: dict = metadata_profile) -> dict:

#     preprocessed_metadata_batch = {}
#     for m in raw_metadata_batch.values():
#         pmm = preprocess_metadata(m, metadata_profile)
#         pmm["Namespace"] = metadata_profile["Namespace"]
#         preprocessed_metadata_batch[pmm["Mnemonic"]] = pmm
#     return preprocessed_metadata_batch


def preprocess_metadata_batch(
    raw_metadata_file_path: str,
    raw_metadata_file_reader: Callable[[str], dict] = read_from_json_default,
    metadata_profile: dict = metadata_profile_mnemonic_rich,
) -> dict:
    preprocessable_metadata_batch = raw_metadata_file_reader(raw_metadata_file_path)

    preprocessed_metadata_batch = {}
    for m in preprocessable_metadata_batch.values():
        pmm = preprocess_metadata(m, metadata_profile)
        pmm["Namespace"] = metadata_profile["Namespace"]
        preprocessed_metadata_batch[pmm["Mnemonic"]] = pmm
    return preprocessed_metadata_batch


def test_short():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    project_folder_path = currentFolder + "/tasks/project_test/"
    raw_metadata_path = currentFolder + "/data_store/mnemonic_rich/items_test_short.json"
    with open(raw_metadata_path, "r") as json_file:
        raw_mnemonic_metadata = json.load(json_file)

    preprocessed_metadata_batch = preprocess_metadata_batch(raw_mnemonic_metadata)
    with open(project_folder_path + "/preprocessed_metadata_batch.json", "w") as json_file:
        json.dump(preprocessed_metadata_batch, json_file, indent=4)


def test_onsite():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    project_folder_path = currentFolder + "/tasks/project_onsite/"
    if not os.path.exists(project_folder_path):
        os.makedirs(project_folder_path)

    raw_metadata_path = currentFolder + "/data_store/mnemonic_onsite/00001_test.xml"
    preprocessed_metadata_batch = preprocess_metadata_batch(
        raw_metadata_path, read_from_xml_onsite, metadata_profile_onsite
    )
    with open(project_folder_path + "/preprocessed_metadata_batch.json", "w") as json_file:
        json.dump(preprocessed_metadata_batch, json_file, indent=4)


if __name__ == "__main__":
    test_onsite()
