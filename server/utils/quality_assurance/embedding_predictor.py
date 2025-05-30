import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent

sys.path.append(str(project_root))

import server.utils.configs.globals_config as glb  # noqa
import server.utils.filter as ft  # noqa


# For e5 models, adding role-specific prefixes significantly improves performance
def to_query(text):
    return f"query: {text}"


def to_passage(text):
    return f"passage: {text}"


# Load the embedding model (use "cpu" if you don't have a GPU)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using device:", device)
model = SentenceTransformer("intfloat/e5-large-v2", device="cuda:0")


def label_dict_to_label_list(label_extraContent: dict) -> list[tuple[str, str]]:
    labels = []
    for key, meta in (label_extraContent).items():
        labels.append((key, str(meta)))
    return labels


# Kick out zzz#TBD, OutOfSet, Uncertain
kick_out_keys = ["zzz#TBD", "OutOfSetPrototypeData", "UncertainPrototypeData", "OutOfSetQuantity", "UncertainQuantity", "OutOfSetUnit", "UncertainUnit"]
quantity_fullList_extraContent = glb.quantity_fullList_extraContent.copy()
unit_fullList_extraContent = glb.unit_fullList_extraContent.copy()
prototypeData_fullList_extraContent = glb.prototypeData_fullList_extraContent.copy()

for k in kick_out_keys:
    quantity_fullList_extraContent.pop(k, None)
    unit_fullList_extraContent.pop(k, None)
    prototypeData_fullList_extraContent.pop(k, None)

labels_quantity = label_dict_to_label_list(quantity_fullList_extraContent)
labels_unit = label_dict_to_label_list(unit_fullList_extraContent)
labels_prototypedata = label_dict_to_label_list(prototypeData_fullList_extraContent)

# 1) Encode label passages offline (one-time). We include the label name to strengthen identity.
label_texts_quantity = [to_passage(desc) for _, desc in labels_quantity]
label_emb_quantity = model.encode(label_texts_quantity, normalize_embeddings=True, convert_to_tensor=True)

label_texts_unit = [to_passage(desc) for _, desc in labels_unit]
label_emb_unit = model.encode(label_texts_unit, normalize_embeddings=True, convert_to_tensor=True)

label_texts_prototypedata = [to_passage(desc) for _, desc in labels_prototypedata]
label_emb_prototypedata = model.encode(label_texts_prototypedata, normalize_embeddings=True, convert_to_tensor=True)


def embedding_predict(sample: Union[dict, str], label_list: list, embedding_matrix: torch.Tensor, top_k=5):
    query = str(sample)
    q_emb = model.encode([to_query(query)], normalize_embeddings=True, convert_to_tensor=True)

    # Cosine similarity + Top-k selection
    scores = util.cos_sim(q_emb, embedding_matrix)[0]  # shape: [num_labels]
    idx = torch.argsort(scores, descending=True)[:top_k]
    cands = {label_list[i][0]: float(scores[i]) for i in idx}
    return cands


query = {"Mnemonic": "SPPA", "Description": "Standpipe pressure.", "Unit": "Pa"}
result = embedding_predict(query, labels_quantity, label_emb_quantity, top_k=5)
print(result)


def softmax_dict(d: dict) -> tuple:
    # Step 1: softmax
    values = np.array(list(d.values()), dtype=float)
    exp_values = np.exp(values - np.max(values))
    softmax_values = exp_values / np.sum(exp_values)
    softmaxed = dict(zip(d.keys(), softmax_values))

    # Step 2:return key with max
    max_val = np.max(softmax_values)
    for k, v in softmaxed.items():
        if np.isclose(v, max_val):
            max_key = k
            break
    return softmaxed, max_key


def get_embedding_recognition_results(query: dict, top_k: int = 5) -> dict:
    result_quantity = embedding_predict(
        sample=query,
        label_list=labels_quantity,
        embedding_matrix=label_emb_quantity,
        top_k=top_k,
    )
    result_unit = embedding_predict(
        sample=query,
        label_list=labels_unit,
        embedding_matrix=label_emb_unit,
        top_k=top_k,
    )
    result_prototypeData = embedding_predict(
        sample=query,
        label_list=labels_prototypedata,
        embedding_matrix=label_emb_prototypedata,
        top_k=top_k,
    )
    recognition_results = {}
    recognition_results.update(
        {
            "Raw_content": str(query),
            "Quantity_class": softmax_dict(result_quantity)[1],
            "Quantity_class_candidates": softmax_dict(result_quantity)[0],
            "Unit_class": softmax_dict(result_unit)[1],
            "Unit_class_candidates": softmax_dict(result_unit)[0],
            "PrototypeData_class": softmax_dict(result_prototypeData)[1],
            "PrototypeData_class_candidates": softmax_dict(result_prototypeData)[0],
        }
    )
    return recognition_results


def test():
    # -----------------------
    # Minimal working example
    # -----------------------

    sample = {"Mnemonic": "SPP", "Description": "SPP is a stand pipe pressure. It has a known accuracy of approx. ±5%.", "Unit": "bar"}

    recognition_ranking = get_embedding_recognition_results(sample, 2)
    print(recognition_ranking)


if __name__ == "__main__":
    test()
