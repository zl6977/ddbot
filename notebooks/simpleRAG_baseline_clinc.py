import json
import sys
import logging
from pathlib import Path

import numpy as np
import yaml
import pandas as pd


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.append(str(project_root))

import server.utils.configs.globals_config as glb  # noqa
import server.utils.filter as ft  # noqa
import server.utils.chat_restapi as llm  # noqa
import server.utils.configs.log_config as log_config  # noqa

logger = logging.getLogger(__name__)
log_config.configure_logger(glb.default_log_path, "w", True, logging.INFO)

# dataset_path = project_root / "data_store/test_data/clinc_oos/small/validation-00000-of-00001.parquet"  # path to small/validation-00000-of-00001.parquet
dataset_path = project_root / "data_store/test_data/clinc_oos/small/reduced_validation_set.parquet"  # path to reduced dataset
df = pd.read_parquet(dataset_path)
true_label_dict = dict(zip(df["text"], df["intent"]))
# print(true_label_dict)

labels_path = project_root / "data_store/test_data/clinc_oos/labels.yaml"
with open(labels_path, "r") as file:
    labels_obj = yaml.safe_load(file)
full_labels = labels_obj.get("names")

labels = {str(i): full_labels[str(i)] for i in range(30)}
labels.update({"42": "oos"})
splited_labels = [label.replace("_", " ") for label in labels.values()]
print(splited_labels)


def softmax_dict(d: dict, temperature: float = 1.0) -> tuple:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    values = np.array(list(d.values()), dtype=float)
    # subtract max for numerical stability, then divide by T
    scaled = (values - np.max(values)) / temperature
    exp_values = np.exp(scaled)
    softmax_values = exp_values / np.sum(exp_values)

    softmaxed = dict(zip(d.keys(), softmax_values))

    # key with max probability
    max_val = np.max(softmax_values)
    for k, v in softmaxed.items():
        if np.isclose(v, max_val):
            max_key = k
            break

    return softmaxed, max_key


query_files = [
    project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+1+1+1+00001.json",
    project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+1+1+00001.json",
    project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+2+1+00001.json",
]


def simpleRAG_predict(sample, label_list):
    prompt_template = """You are a classifier.
You will receive a query sentence and a list of labels. Your task is to select one label that best describes the query sentence.
If no suitable label exists, return "oos".
Input:
Query sentence:
<query_sentence>
Labels:
<labels>
Output:
Only the single chosen label — no explanations or additional text.
    """
    prompt = prompt_template.replace("<query_sentence>", str(sample))
    prompt = prompt.replace("<labels>", str(label_list))
    # data_config = {"logit_bias": {8459: -100, 10843: -100, 6: 50, 27384: 50, 21757: 50, 1366: 50, 8505: 50}}
    data_config = {
        # "model": "deepseek/deepseek-r1:free",
        # "reasoning_effort": "medium",
        "temperature": 0.1,
    }
    model = "gpt-4o-mini"
    response = llm.chat_with_llm(prompt, model=model, user_config=data_config)
    result = llm.result_extractor(response, model=model)
    # result_list = response_data["message"]["content"].split(",")
    logger.info(f"prompt: {prompt}")
    return result["content"]


def run_on_clinc():
    global df, labels
    result_clinc = {}
    for i in range(len(df["text"].tolist())):
        q_str = df["text"].values[i]
        intent = str(true_label_dict[q_str])
        tmp = {"true_label": labels[intent], "candidates": simpleRAG_predict(q_str, list(labels.values()))}
        result_clinc.update({q_str: tmp})
        logger.info(f"{i}/{len(df['text'])} candidates {tmp['candidates']}")

    Path("simpleRAG").mkdir(parents=True, exist_ok=True)
    with open(current_dir / "simpleRAG/recognition_results_clinc-2.json", "w", encoding="utf-8") as f:
        json.dump(result_clinc, f, ensure_ascii=False, indent=2)


# def run_on_witsml():
#     query_objs = []
#     for i in range(len(query_files)):
#         with open(query_files[i], "r", encoding="utf-8") as f:
#             data = json.load(f)
#             query_tmp = {}
#             for j in range(len(data)):
#                 mnemonic = data[j]["Mnemonic"]
#                 data[j].pop("true label PrototypeData")
#                 data[j].pop("true label Unit")
#                 data[j].pop("true label Quantity")
#                 query_tmp.update({mnemonic: data[j]})
#             query_objs.append(query_tmp)
#             filtered, rejected = ft.filter_metadata(query_objs[i])
#             print("filtered count:", len(filtered))
#             print("rejected count:", len(rejected))

#         recognition_results = {}
#         for k, v in filtered.items():
#             result_quantity = simpleRAG_predict(
#                 sample=v,
#                 label_list=labels_quantity,
#                 embedding_matrix=label_emb_quantity,
#                 top_k=5,
#             )
#             result_unit = simpleRAG_predict(
#                 sample=v,
#                 label_list=labels_unit,
#                 embedding_matrix=label_emb_unit,
#                 top_k=5,
#             )
#             result_prototypeData = simpleRAG_predict(
#                 sample=v,
#                 label_list=labels_prototypedata,
#                 embedding_matrix=label_emb_prototypedata,
#                 top_k=5,
#             )
#             softmax_temperature = 0.005
#             recognition_results.update(
#                 {
#                     k: {
#                         "Raw_content": str(v),
#                         "Quantity_class": softmax_dict(result_quantity, softmax_temperature)[1],
#                         "Quantity_class_candidates": softmax_dict(result_quantity, softmax_temperature)[0],
#                         "Unit_class": softmax_dict(result_unit, softmax_temperature)[1],
#                         "Unit_class_candidates": softmax_dict(result_unit, softmax_temperature)[0],
#                         "PrototypeData_class": softmax_dict(result_prototypeData, softmax_temperature)[1],
#                         "PrototypeData_class_candidates": softmax_dict(result_prototypeData, softmax_temperature)[0],
#                     }
#                 }
#             )

#             Path("simpleRAG").mkdir(parents=True, exist_ok=True)
#             with open(current_dir / "simpleRAG/recognition_results_clinc.json", "w", encoding="utf-8") as f:
#                 json.dump(recognition_results, f, ensure_ascii=False, indent=2)


def test():
    sample = "does mcdonald's have good reviews"
    label_list = list(labels.values())
    simpleRAG_predict(sample, label_list)


if __name__ == "__main__":
    run_on_clinc()
