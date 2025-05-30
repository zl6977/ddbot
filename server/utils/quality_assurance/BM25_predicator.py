import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent

sys.path.append(str(project_root))

import server.utils.configs.globals_config as glb  # noqa
import server.utils.filter as ft  # noqa
# -----------------------------
# Simple, dependency-free BM25
# -----------------------------


def tokenize(s: str):
    _word_re = re.compile(r"[A-Za-z0-9%/±]+")
    return [t.lower() for t in _word_re.findall(s or "")]


def concat_with_weights(fields, weights):
    """
    fields: dict[str, list[str] | str]  (already tokenized or raw)
    weights: dict[str, float]
    returns: list[str] tokens with repetition to reflect weights
    """
    bag = []
    for fname, content in fields.items():
        w = weights.get(fname, 1.0)
        if content is None:
            continue
        if isinstance(content, str):
            toks = tokenize(content)
        else:
            # list[str]
            toks = []
            for x in content:
                toks.extend(tokenize(x))
        # repeat tokens by weight's integer part, keep fractional via probabilistic drop
        k = int(w)
        frac = w - k
        if k > 0:
            bag.extend(toks * k)
        if frac > 0:
            # lightweight fractional weighting via sampling-free rounding
            # (multiply some tokens by +1)
            extra = round(frac * len(toks))
            bag.extend(toks[:extra])
    return bag


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """
        docs: list[list[str]] tokenized docs
        """
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        # df and idf
        self.df = Counter()
        for d in docs:
            for t in set(d):
                self.df[t] += 1
        # Using BM25+ style log idf (Okapi variant)
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        # term frequencies
        self.tf = [Counter(d) for d in docs]

    def score(self, q_tokens):
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
            tf_i = self.tf[i]
            s = 0.0
            for t in q_tokens:
                if t not in self.idf:
                    continue
                f = tf_i.get(t, 0)
                if f == 0:
                    continue
                s += self.idf[t] * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores


def bm25_predict(sample_dict, labels_dict, build_query_tokens_func, build_label_doc_func, k1=1.5, b=0.75, top_k=5):
    label_names = list(labels_dict.keys())
    docs = [build_label_doc_func(labels_dict[name]) for name in label_names]
    model = BM25(docs, k1=k1, b=b)
    q = build_query_tokens_func(sample_dict)
    scores = model.score(q)
    ranked = sorted(zip(label_names, scores), key=lambda x: x[1], reverse=True)
    result = {}
    for i in range(top_k):
        result.update({ranked[i][0]: ranked[i][1]})
    return result


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


def build_query_tokens(sample):
    QUERY_FIELD_WEIGHTS = {
        "Mnemonic": 4.0,
        "Description": 3.0,
        "Unit": 2.0,  # units are helpful signals (e.g., bar, m/s)
    }
    fields = {
        "Mnemonic": sample.get("Mnemonic", ""),
        "Description": sample.get("Description", ""),
        "Unit": sample.get("Unit", ""),
    }
    return concat_with_weights(fields, QUERY_FIELD_WEIGHTS)


def build_label_doc_prototypedata(label_meta):
    LABEL_FIELD_WEIGHTS = {
        "mnemonic": 4.0,  # exact code like "SPP", "ROP" is very telling
        "comment": 3.0,  # human description
        "base_quantities": 1.5,  # PressureQuantity, VelocityQuantity, etc.
        "measurable_quantity": 1.0,
        "common_mnemonics": 4.0,  # other aliases/codes
    }
    fields = {
        "mnemonic": label_meta.get("ddhub:PrototypeData"),
        "comment": " ".join(label_meta.get("rdfs:comment", [])),
        "base_quantities": label_meta.get("ddhub:IsOfBaseQuantity", []),
        "measurable_quantity": label_meta.get("ddhub:IsOfMeasurableQuantity"),
        "common_mnemonics": label_meta.get("zzz:commonMnemonics", []),
    }
    return concat_with_weights(fields, LABEL_FIELD_WEIGHTS)


def build_label_doc_quantity(label_meta):
    LABEL_FIELD_WEIGHTS = {
        "mnemonic": 4.0,  # exact code like "SPP", "ROP" is very telling
        "comment": 3.0,  # human description
        "quantityHasUnit": 2.0,  # PressureQuantity, VelocityQuantity, etc.
        "prototypeData": 1.0,
    }
    fields = {
        "mnemonic": label_meta.get("ddhub:Quantity"),
        "comment": " ".join(label_meta.get("rdfs:comment", [])),
        "quantityHasUnit": label_meta.get("zzz:QuantityHasUnit", []),
        "prototypeData": label_meta.get("zzz:PrototypeData", []),
    }
    return concat_with_weights(fields, LABEL_FIELD_WEIGHTS)


def build_label_doc_unit(label_meta):
    LABEL_FIELD_WEIGHTS = {
        "mnemonic": 4.0,  # exact code like "SPP", "ROP" is very telling
        "comment": 3.0,  # human description
        "isUnitForQuantity": 1.5,  # PressureQuantity, VelocityQuantity, etc.
        "common_mnemonics": 4.0,  # other aliases/codes
    }
    fields = {
        "mnemonic": label_meta.get("ddhub:PrototypeData"),
        "comment": " ".join(label_meta.get("rdfs:comment", [])),
        "isUnitForQuantity": label_meta.get("ddhub:IsUnitForQuantity", []),
        "common_mnemonics": label_meta.get("zzz:commonMnemonics", []),
    }
    return concat_with_weights(fields, LABEL_FIELD_WEIGHTS)


def get_bm25_recognition_results(query: dict, top_k: int = 5) -> dict:
    """
    Predicts Quantity, Unit, and PrototypeData classes for a single query using BM25 scoring.

    :param query: A dictionary containing query metadata (e.g., Mnemonic, Description, Unit).
    :param mnemonic: The key/identifier for the query result (e.g., the Mnemonic for the sample).
    :return: A dictionary containing the recognition results keyed by the provided mnemonic.
    """
    # Kick out zzz#TBD, OutOfSet, Uncertain
    kick_out_keys = ["zzz#TBD", "OutOfSetPrototypeData", "UncertainPrototypeData", "OutOfSetQuantity", "UncertainQuantity", "OutOfSetUnit", "UncertainUnit"]
    for k in kick_out_keys:
        glb.quantity_fullList_extraContent.pop(k, None)
        glb.unit_fullList_extraContent.pop(k, None)
        glb.prototypeData_fullList_extraContent.pop(k, None)

    recognition_results = {}
    result_quantity = bm25_predict(
        sample_dict=query,
        labels_dict=glb.quantity_fullList_extraContent,
        build_query_tokens_func=build_query_tokens,
        build_label_doc_func=build_label_doc_quantity,
        top_k=top_k,
    )
    result_unit = bm25_predict(
        sample_dict=query,
        labels_dict=glb.unit_fullList_extraContent,
        build_query_tokens_func=build_query_tokens,
        build_label_doc_func=build_label_doc_unit,
        top_k=top_k,
    )
    result_prototypeData = bm25_predict(
        sample_dict=query,
        labels_dict=glb.prototypeData_fullList_extraContent,
        build_query_tokens_func=build_query_tokens,
        build_label_doc_func=build_label_doc_prototypedata,
        top_k=top_k,
    )
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

    # labels_obj = {
    #     "ROP": {
    #         "ddhub:PrototypeData": "ROP",
    #         "rdfs:comment": ["Rate of penetration, the speed at which the drill bit advances."],
    #         "ddhub:IsOfMeasurableQuantity": "RateOfPenetrationDrillingQuantity",
    #         "ddhub:IsOfBaseQuantity": ["UncertainQuantity", "OutOfSetQuantity", "VelocityQuantity"],
    #         "zzz:commonMnemonics": ["ROP"],
    #     },
    #     "SPP": {
    #         "ddhub:PrototypeData": "SPP",
    #         "rdfs:comment": ["Standpipe pressure, the pressure in the drill pipe."],
    #         "ddhub:IsOfMeasurableQuantity": "PressureDrillingQuantity",
    #         "ddhub:IsOfBaseQuantity": ["UncertainQuantity", "OutOfSetQuantity", "PressureQuantity"],
    #         "zzz:commonMnemonics": ["SPP"],
    #     },
    # }
    recognition_ranking = get_bm25_recognition_results(sample, 2)
    print(recognition_ranking)

if __name__ == "__main__":
    test()
