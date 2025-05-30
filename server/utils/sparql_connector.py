import json
import logging
import os

import requests
from rdflib import Graph
from rdflib.query import ResultRow

logger = logging.getLogger(__name__)
utils_folder = os.path.dirname(os.path.realpath(__file__))
KB_ttl_path = utils_folder + "/../../data_store/DDHub_model/DWISVocabulary_merged-202504.ttl"
SPARQL_endpoint = "http://localhost:3030/DWISVocabulary/"


def query_from_KB_ttl(ttl_file_path: str, query: str) -> dict:
    g = Graph()
    g.parse(ttl_file_path, format="ttl")

    # Apply the query to the graph and iterate through results
    response = g.query(query)
    result = {}
    data = []
    for row in response:
        col = []
        if isinstance(row, ResultRow):
            for c in row:
                col.append(str(c))
            data.append(col)
    if response.vars is not None:
        result = {
            "header": [var.n3()[1:] for var in response.vars],
            "data": data,
        }
    return result


def query_from_KB(endpoint: str, query: str) -> dict:
    response = requests.post(
        endpoint,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
    )
    response_dict = response.json()
    header = response_dict["head"]["vars"]
    # header = [var.n3()[1:] for var in response_dict["head"]["vars"]]
    data = []
    for body in response_dict["results"]["bindings"]:
        values = []
        for key in body:
            values.append(body[key]["value"])
        data.append(values)
    result = {
        "header": header,
        "data": data,
    }
    return result


def make_result_table(query_result: dict, no_namespace: bool = True) -> list:
    result_table = []
    result_data = query_result["data"]
    for r in result_data:
        print(r)
    return result_table


def parse_keywords(input_string):
    keywords = input_string.split("?")[1:]
    keywords = [keyword.strip() for keyword in keywords]
    result = {keyword: index for index, keyword in enumerate(keywords)}
    return result


def separate_list_item(input_list: list) -> list:
    output_set = set()
    for item in input_list:
        # 按逗号分隔并去除空格
        list_tmp = item.split(",")
        output_set.update(list_tmp)
    return list(output_set)


"""
To add new SPARQL query module:
1. Write a "make_sparql_xxx()" function to generate SPARQL string.
2. Write a controlling function to cover: SPARQL string generation, query_from_KB, result extraction.
"""


def make_querytring_MQuantity_relatedTo_ProtytypeData(PrototypeData: str) -> str:
    query_template = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?MeasurableQuantity
WHERE { 
    ddhub:<PrototypeData> rdf:type owl:Class ;
            rdfs:subClassOf [ 
                rdf:type owl:Restriction ;
                owl:onProperty ddhub:IsOfMeasurableQuantity ;
                owl:allValuesFrom ?MeasurableQuantity ].
}
    """
    query = query_template.replace("<PrototypeData>", PrototypeData)
    return query


def query_MQuantity(ttl_file_path: str, PrototypeData: str):
    query = make_querytring_MQuantity_relatedTo_ProtytypeData(PrototypeData)
    response = query_from_KB_ttl(ttl_file_path, query)
    # response = query_from_KB(SPARQL_endpoint, query)
    # print(response)
    MeasurableQuantity = response["data"][0]
    return MeasurableQuantity


## --- PrototypeData extra content
def make_queryStr_PrototypeData_comment():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?PrototypeData ?comment
WHERE {
    ?PrototypeData rdfs:subClassOf ddhub:PrototypeData ;
                  rdfs:comment ?comment .
    FILTER(STR(?comment) != "")
}
"""

def make_queryStr_PrototypeData_mnemonics():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?PrototypeData ?commonMnemonics
WHERE {
    ?PrototypeData rdfs:subClassOf ddhub:PrototypeData ;
                  zzz:commonMnemonics ?commonMnemonics .
    FILTER(STR(?commonMnemonics) != "")
}
"""

def make_queryStr_PrototypeData_MQ():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?PrototypeData ?MeasurableQuantity
WHERE {
    ?PrototypeData rdfs:subClassOf [
        rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfMeasurableQuantity ;
        owl:allValuesFrom ?MeasurableQuantity
    ] .
}
"""

def make_queryStr_PrototypeData_Quantity():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?PrototypeData ?Quantity
WHERE {
    ?PrototypeData rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfMeasurableQuantity ;
        owl:allValuesFrom ?mq .

    ?mq rdfs:subClassOf ?restriction2 .
    ?restriction2 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfBaseQuantity ;
        owl:allValuesFrom ?Quantity .
    FILTER(?Quantity != ddhub:Quantity)
}
"""


def generate_PrototypeData_fullList_extraContent(source_ttl_path: str) -> dict:
    """Retrieve PrototypeData extra content via three lightweight queries and merge.

    Returns schema identical to previous implementation to preserve callers:
        PrototypeData -> {
            "ddhub:PrototypeData": str,
            "rdfs:comment": [str, ...],
            "ddhub:IsOfMeasurableQuantity": str | "None",
            "ddhub:IsOfBaseQuantity": [str, ..],
            "zzz:commonMnemonics": [str, ...] (optional)
        }
    """
    q_comments = query_from_KB_ttl(source_ttl_path, make_queryStr_PrototypeData_comment())
    q_mnemonics = query_from_KB_ttl(source_ttl_path, make_queryStr_PrototypeData_mnemonics())
    q_pd_mq = query_from_KB_ttl(source_ttl_path, make_queryStr_PrototypeData_MQ())
    q_pd_q = query_from_KB_ttl(source_ttl_path, make_queryStr_PrototypeData_Quantity())

    fullList: dict[str, dict] = {}

    def ensure(pdid):
        if pdid not in fullList:
            fullList[pdid] = {
                "ddhub:PrototypeData": pdid,
                "rdfs:comment": [],
                "ddhub:IsOfMeasurableQuantity": "UncertainMeasurableQuantity",
                "ddhub:IsOfBaseQuantity": ["UncertainQuantity", "OutOfSetQuantity"],
                "zzz:commonMnemonics": [],
            }

    for r in q_comments["data"]:
        pd = r[0].split("/")[-1]
        comment = r[1]
        ensure(pd)
        if comment not in fullList[pd]["rdfs:comment"]:
            fullList[pd]["rdfs:comment"].append(comment)

    for r in q_mnemonics["data"]:
        pd = r[0].split("/")[-1]
        cmn = r[1] if len(r) > 1 else None
        ensure(pd)
        if cmn and cmn not in fullList[pd]["zzz:commonMnemonics"]:
            fullList[pd]["zzz:commonMnemonics"].append(cmn)

    for r in q_pd_mq["data"]:
        pd = r[0].split("/")[-1]
        mq = r[1].split("/")[-1]
        ensure(pd)
        if fullList[pd]["ddhub:IsOfMeasurableQuantity"] == "UncertainMeasurableQuantity":
            fullList[pd]["ddhub:IsOfMeasurableQuantity"] = mq

    for r in q_pd_q["data"]:
        pd = r[0].split("/")[-1]
        q = r[1].split("/")[-1]
        ensure(pd)
        if q not in fullList[pd]["ddhub:IsOfBaseQuantity"]:
            fullList[pd]["ddhub:IsOfBaseQuantity"].append(q)

    return fullList


## --- Unit extra content

def make_queryStr_Unit_comment():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?Unit ?comment
WHERE {
    ?Unit rdfs:subClassOf ddhub:Unit ;
          rdfs:comment ?comment .
    FILTER(STR(?comment) != "")
}
"""

def make_queryStr_Unit_mnemonics():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
PREFIX zzz: <http://ddhub.demo/zzz#>
SELECT DISTINCT ?Unit ?commonMnemonics
WHERE {
    ?Unit rdfs:subClassOf ddhub:Unit ;
          zzz:commonMnemonics ?commonMnemonics .
    FILTER(STR(?commonMnemonics) != "")
}
"""

def make_queryStr_Unit_quantities():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?Unit ?Quantity
WHERE {
    ?Unit rdfs:subClassOf [
        rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsUnitForQuantity ;
        owl:allValuesFrom ?allValuesClass
    ] .
    ?allValuesClass owl:unionOf ?unionList .
    ?unionList rdf:rest*/rdf:first ?Quantity .
    FILTER(?Quantity != ddhub:Quantity)
}
"""


def generate_Unit_fullList_extraContent(source_ttl_path: str) -> dict:
    """Retrieve Unit extra content via two lightweight queries.

    Output schema unchanged:
        Unit -> {
            "ddhub:Unit": str,
            "rdfs:comment": [str, ...],
            "ddhub:IsUnitForQuantity": [str, ...],
            "zzz:commonMnemonics": [str, ...] (optional)
        }
    """
    q_comments = query_from_KB_ttl(source_ttl_path, make_queryStr_Unit_comment())
    q_mnemonics = query_from_KB_ttl(source_ttl_path, make_queryStr_Unit_mnemonics())
    q_quantities = query_from_KB_ttl(source_ttl_path, make_queryStr_Unit_quantities())

    fullList = {}

    def ensure(uid):
        if uid not in fullList:
            fullList[uid] = {
                "ddhub:Unit": uid,
                "rdfs:comment": [],
                "ddhub:IsUnitForQuantity": ["UncertainQuantity", "OutOfSetQuantity"],
                "zzz:commonMnemonics": [],
            }

    for r in q_comments["data"]:
        unit = r[0].split("/")[-1]
        comment = r[1]
        ensure(unit)
        if comment not in fullList[unit]["rdfs:comment"]:
            fullList[unit]["rdfs:comment"].append(comment)

    for r in q_mnemonics["data"]:
        unit = r[0].split("/")[-1]
        mnemo = r[1]
        ensure(unit)
        if mnemo not in fullList[unit]["zzz:commonMnemonics"]:
            fullList[unit]["zzz:commonMnemonics"].append(mnemo)

    for r in q_quantities["data"]:
        unit = r[0].split("/")[-1]
        quantity = r[1].split("/")[-1]
        ensure(unit)
        if quantity not in fullList[unit]["ddhub:IsUnitForQuantity"]:
            fullList[unit]["ddhub:IsUnitForQuantity"].append(quantity)

    return fullList

# ---- Quantity Data Extra Content

def make_queryStr_Quantities_with_comments():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?Quantity ?comment
WHERE {
    ?Quantity rdfs:subClassOf ddhub:Quantity .
    ?Quantity rdfs:comment ?comment .
    FILTER(STR(?comment) != "")
}
"""

def make_queryStr_Units_for_Quantities():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?Quantity ?Unit
WHERE {
    ?Unit rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsUnitForQuantity ;
        owl:allValuesFrom ?allValuesClass .

    ?allValuesClass owl:unionOf ?unionList .
    ?unionList rdf:rest*/rdf:first ?Quantity .
    FILTER(?Quantity != ddhub:Quantity)
}
"""

def make_queryStr_PrototypeData_for_Quantities():
    return """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ddhub: <http://ddhub.no/>
SELECT DISTINCT ?Quantity ?PrototypeData
WHERE {
    ?PrototypeData rdfs:subClassOf ?restriction1 .
    ?restriction1 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfMeasurableQuantity ;
        owl:allValuesFrom ?mq .

    ?mq rdfs:subClassOf ?restriction2 .
    ?restriction2 rdf:type owl:Restriction ;
        owl:onProperty ddhub:IsOfBaseQuantity ;
        owl:allValuesFrom ?Quantity .
    FILTER(?Quantity != ddhub:Quantity)
}
"""

def generate_Quantity_fullList_extraContent(source_ttl_path: str) -> dict:
    q_comments = query_from_KB_ttl(source_ttl_path, make_queryStr_Quantities_with_comments())
    q_units = query_from_KB_ttl(source_ttl_path, make_queryStr_Units_for_Quantities())
    q_protos = query_from_KB_ttl(source_ttl_path, make_queryStr_PrototypeData_for_Quantities())

    fullList: dict[str, dict] = {}

    def ensure(qid: str):
        if qid not in fullList:
            fullList[qid] = {
                "ddhub:Quantity": qid,
                "rdfs:comment": [],
                "zzz:QuantityHasUnit": ["UncertainUnit", "OutOfSetUnit"],
                "zzz:PrototypeData": ["UncertainPrototypeData", "OutOfSetPrototypeData"],
            }
            
    # Comments
    for r in q_comments["data"]:
        quantity = r[0].split("/")[-1]
        comment = r[1]
        ensure(quantity)
        if comment not in fullList[quantity]["rdfs:comment"]:
            fullList[quantity]["rdfs:comment"].append(comment)

    # Units
    for r in q_units["data"]:
        quantity = r[0].split("/")[-1]
        unit = r[1].split("/")[-1]
        ensure(quantity)
        lst = fullList[quantity]["zzz:QuantityHasUnit"]
        if unit not in lst:
            lst.append(unit)

    # PrototypeData
    for r in q_protos["data"]:
        quantity = r[0].split("/")[-1]
        proto = r[1].split("/")[-1]
        ensure(quantity)
        lst = fullList[quantity]["zzz:PrototypeData"]
        if proto not in lst:
            lst.append(proto)

    return fullList


# -------------


def test_ask_MeasurableQuantity():
    ttl_file_path = KB_ttl_path
    r = query_MQuantity(ttl_file_path, "HookLoad")
    print(r)


def test_retrieve_context():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    ttl_file_path = KB_ttl_path

    context = generate_PrototypeData_fullList_extraContent(ttl_file_path)
    with open(currentFolder + "/data_store/DDHub_model/PrototypeData_fullList_extraContent.json", "w") as json_file:
        json.dump(context, json_file, indent=4)



if __name__ == "__main__":
    # test_ask_MeasurableQuantity()
    test_retrieve_context()
