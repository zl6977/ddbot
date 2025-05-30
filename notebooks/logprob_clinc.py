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
import server.utils.validation.probability_based_selection as pbs  # noqa


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


# query_files = [
#     project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+1+1+1+00001.json",
#     project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+1+1+00001.json",
#     project_root / "data_store/test_data/new_labels/extracted_json/9-F-9_A+1+log+2+2+1+00001.json",
# ]


def logprob_predict(sample, label_list):
    prompt_template = """You are a classifier.
You will receive a query sentence and a list of labels. Your task is to select one label that best describes the query sentence.
If no suitable label exists, choose "oos".
Input:
Query sentence: 
<query_sentence>
Labels:
<labels>
Output:
Only the single chosen label index — no explanations or additional text.
    """

    # Prepare key-value pairs to populate the prompt template
    kvPairs = {"<query_sentence>": str(sample)}
    # Rank options by LLM probabilities to get the aggregated distribution
    probs_distribution_agg = pbs.rank_options_tournament(
        prompt_template=prompt_template,
        options=label_list,
        options_placeholder="<labels>",
        option_explanations={},
        option_explanations_placeholder="N.A.",
        kvPairs=kvPairs,
        rounds=1,
        pool_size=16,
    )
    # probs_distribution_agg is like [{"Option A": {"average": 0.805, "history": [0.8, 0.81]}}, ...]
    probs_distribution = {list(d.keys())[0]: list(d.values())[0]["average"] for d in probs_distribution_agg}
    logger.debug(f"Probability distribution from LLM: {probs_distribution}")
    return probs_distribution


def run_on_clinc():
    global df, labels
    result_clinc = {}
    for i in range(len(df["text"].tolist())):
        q_str = df["text"].values[i]
        intent = str(true_label_dict[q_str])
        tmp = {"true_label": labels[intent], "candidates": logprob_predict(q_str, list(labels.values()))}
        result_clinc.update({q_str: tmp})
        logger.info(f"{i + 1}/{len(df['text'])} candidates {tmp['candidates']}")

    Path("logprob").mkdir(parents=True, exist_ok=True)
    with open(current_dir / "logprob/recognition_results_clinc.json", "w", encoding="utf-8") as f:
        json.dump(result_clinc, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_on_clinc()
