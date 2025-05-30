import json
from pathlib import Path

import pandas as pd
import yaml

current_dir = Path(__file__).resolve().parent

dataset_path = current_dir / "validation-00000-of-00001.parquet"  # path to small/validation-00000-of-00001.parquet
df = pd.read_parquet(dataset_path)

labels_path = current_dir / "./../labels.yaml"
with open(labels_path, "r") as file:
    labels_obj = yaml.safe_load(file)
labels = labels_obj.get("names")


full_dataset = {}
queries = df["text"].values.tolist()
intent_index = df["intent"].values.tolist()
for i in range(len(df["text"])):
    tmp = {queries[i]: labels[str(intent_index[i])]}
    full_dataset.update(tmp)


def retrieve_related_queries(true_labels, label):
    related_queries = []
    for qry, lbl in true_labels.items():
        if lbl == label:
            related_queries.append(qry)
    return related_queries


selected_labels = []
for label_index in range(30):
    selected_labels.append(labels[str(label_index)])

reduced_dataset = {}
for label in selected_labels:
    related_queries = retrieve_related_queries(full_dataset, label)
    for query_index_in_class in range(10):
        query = related_queries[query_index_in_class]
        tmp = {query: full_dataset[query]}
        reduced_dataset.update(tmp)

oos_queries = retrieve_related_queries(full_dataset, "oos")
for query in oos_queries:
    tmp = {query: full_dataset[query]}
    reduced_dataset.update(tmp)

with open(current_dir / "full_validation_set.json", "w", encoding="utf-8") as f:
    json.dump(full_dataset, f, ensure_ascii=False, indent=2)
with open(current_dir / "reduced_validation_set.json", "w", encoding="utf-8") as f:
    json.dump(reduced_dataset, f, ensure_ascii=False, indent=2)


def find_intent_index(labels, intent_str):
    for index, string in labels.items():
        if string == intent_str:
            return index


reduced_dataset_intent_index = {qry: find_intent_index(labels, intent_str) for qry, intent_str in reduced_dataset.items()}

data = list(reduced_dataset_intent_index.items())
df = pd.DataFrame(data, columns=["text", "intent"])
df.to_parquet(current_dir / "reduced_validation_set.parquet", index=False)
