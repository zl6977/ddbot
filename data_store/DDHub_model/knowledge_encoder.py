import csv
import os

# import json
# import xml.etree.ElementTree as ET
from typing import Callable

import pandas as pd
from rdflib import Graph
# import yaml


def generate_sparql_segment_PrototypeData(entry: list) -> str:
    PrototypeData = entry[0]
    Description = entry[1].replace("\n", "")
    DrillingQuantity = entry[3]
    commonMnemonics = entry[4]

    sparql_string = f"""{PrototypeData} rdf:type owl:Class ;
          rdfs:subClassOf ddhub:PrototypeData ,
                          [ rdf:type owl:Restriction ;
                            owl:onProperty ddhub:IsOfMeasurableQuantity ;
                            owl:allValuesFrom {DrillingQuantity}
                          ] ;
          zzz:commonMnemonics "{commonMnemonics}"@en ;
          rdfs:comment "{Description}"@en .
    """
    sparql_string = sparql_string.replace('zzz:commonMnemonics "nan"@en ;', "")
    return sparql_string


def generate_sparql_segment_DrillingQuantity(entry: list) -> str:
    MeasurableQuantity = entry[0]
    Quantity = entry[1]

    sparql_string = f"""{MeasurableQuantity} rdf:type owl:Class ;
          rdfs:subClassOf ddhub:MeasurableQuantity ,
                          [ rdf:type owl:Restriction ;
                            owl:onProperty ddhub:IsOfBaseQuantity ;
                            owl:allValuesFrom {Quantity}
                          ] .
    """
    return sparql_string


def generate_sparql_segment_Unit(entry: list) -> str:
    Unit = entry[0]
    Quantity = entry[1]
    commonMnemonics = entry[2]

    sparql_string = f"""{Unit} rdf:type owl:Class ;
          zzz:commonMnemonics "{commonMnemonics}"@en ;
          rdfs:subClassOf ddhub:Unit ,
                          [ rdf:type owl:Restriction ;
                            owl:onProperty ddhub:IsUnitForQuantity ;
                            owl:allValuesFrom [ rdf:type owl:Class ;
                                                owl:unionOf ({Quantity})
                                              ]
                          ] .
    """
    sparql_string = sparql_string.replace('zzz:commonMnemonics "nan"@en ;', "")
    return sparql_string


def read_patch_file_csv(patch_file_path: str) -> list:
    patch_content = []
    with open(patch_file_path, newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            patch_content.append(row)
    return patch_content


def read_patch_file_xlsx(patch_file_path: str) -> list:
    # Read the Excel file, skipping the first row
    df = pd.read_excel(patch_file_path, skiprows=0)
    # Convert the DataFrame to a list of lists
    patch_content = df.values.tolist()
    return patch_content


def generate_sparql_file(entry_batch: list, output_path: str, func_for_sparql_segment: Callable[[list], str]) -> str:
    sparql_body = ""
    for entry in entry_batch:
        sparql_body += func_for_sparql_segment(entry) + "\n"

    sparql_string = """prefix : <http://www.semanticweb.org/owl/owlapi/turtle#>
prefix owl: <http://www.w3.org/2002/07/owl#> 
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
prefix xml: <http://www.w3.org/XML/1998/namespace> 
prefix xsd: <http://www.w3.org/2001/XMLSchema#> 
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
prefix ddhub: <http://ddhub.no/> 
prefix zzz: <http://ddhub.demo/zzz#> 
base <http://www.semanticweb.org/owl/owlapi/turtle#> 

INSERT DATA {
<sparql_body>
}
""".replace("<sparql_body>", sparql_body)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(sparql_string)
    return sparql_string


def convert_patch_to_sparql_batch(patch_info_batch: dict):
    for patch_info in patch_info_batch.values():
        patch_file_path = patch_info["patch_file_path"]
        patch_file_reader = patch_info["patch_file_reader"]
        func_for_sparql_segment = patch_info["func_for_sparql_segment"]

        patch_content = patch_file_reader(patch_file_path)
        suffix = patch_file_path.split(".")[-1]
        output_path = patch_file_path.replace(suffix, "sparql")
        generate_sparql_file(patch_content, output_path, func_for_sparql_segment)


def merge_ttl_files(input_ttl_files_path: list, output_ttl_file_path: str):
    merged_graph = Graph()
    for file in input_ttl_files_path:
        merged_graph.parse(file, format="turtle")
    merged_graph.serialize(destination=output_ttl_file_path, format="turtle", encoding="utf-8")

    with open(output_ttl_file_path, "r", encoding="utf-8") as file:
        ttl_string = file.read()

    # print(f"before: {len(ttl_string)}")
    ttl_string = ttl_string.replace(r"\\r", "<Protected>")
    ttl_string = ttl_string.replace(r"\r", "")
    ttl_string = ttl_string.replace("<Protected>", r"\\r")
    # print(f"after: {len(ttl_string)}")

    # with open(output_ttl_file_path, "w", encoding="utf-8") as file:
    #     file.write(ttl_string)

    refined_graph = Graph()
    refined_graph.parse(data=ttl_string, format="turtle")

    # Clean the graph
    sparql_str = '''prefix zzz: <http://ddhub.demo/zzz#> 
DELETE {
  ?s rdfs:comment ""@en .
  ?s rdfs:comment ""@EN .
  ?s rdfs:comment """\n"""@EN .
  ?s zzz:commonMnemonics ""@en .
  ?s zzz:commonMnemonics "" .
  ?s zzz:commonMnemonics "nan"@en .
}
WHERE {
  { ?s rdfs:comment ""@en }
  UNION
  { ?s rdfs:comment ""@EN }
  UNION
  { ?s rdfs:comment """\n"""@EN }
  UNION
  { ?s zzz:commonMnemonics ""@en }
  UNION
  { ?s zzz:commonMnemonics "" }
  UNION
  { ?s zzz:commonMnemonics "nan"@en }
}
'''

    refined_graph.update(sparql_str)
    refined_graph.serialize(destination=output_ttl_file_path, format="turtle", encoding="utf-8")


def apply_sparql_to_KB(base_ttl_path: str, sparql_path_list: list, destination_path: str):
    g = Graph()
    g.parse(base_ttl_path, format="ttl")

    for sf in sparql_path_list:
        with open(sf, "r", encoding="utf-8") as file:
            added_content_sparql = file.read()
        print(f"To process: {sf}")
        g.update(added_content_sparql)
    g.serialize(destination=destination_path, format="ttl", encoding="utf-8")


# Controllers in application layer

# def merge_ttl_files_control():
#     currentFolder = os.path.dirname(os.path.realpath(__file__))
#     input_ttl_files_path = [
#         currentFolder + "/DWISVocabulary_sparql_patched.ttl",
#         currentFolder + "/Units_Quantities.ttl",
#         currentFolder + "/patch_create.ttl",
#     ]
#     output_ttl_file_path = currentFolder + "/DWISVocabulary_merged.ttl"
#     merge_ttl_files(input_ttl_files_path, output_ttl_file_path)


def apply_knowledge_patches_to_KB():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    # Generate SPARQL patch
    patch_info_batch = {
        "PrototypeData": {
            "patch_file_path": currentFolder + "/Patch_PrototypeData.xlsx",
            "patch_file_reader": read_patch_file_xlsx,
            "func_for_sparql_segment": generate_sparql_segment_PrototypeData,
        },
        "DrillingQuantity": {
            "patch_file_path": currentFolder + "/Patch_DrillingQuantity.xlsx",
            "patch_file_reader": read_patch_file_xlsx,
            "func_for_sparql_segment": generate_sparql_segment_DrillingQuantity,
        },
        "Unit": {
            "patch_file_path": currentFolder + "/Patch_Unit.xlsx",
            "patch_file_reader": read_patch_file_xlsx,
            "func_for_sparql_segment": generate_sparql_segment_Unit,
        },
    }
    convert_patch_to_sparql_batch(patch_info_batch)

    # Write the updated turtle file
    base_ttl_path = currentFolder + "/DWISVocabulary-20250219.ttl"
    sparql_path_list = []
    for patch_info in patch_info_batch.values():
        patch_file_path = patch_info["patch_file_path"]
        suffix = patch_file_path.split(".")[-1]
        output_path = patch_file_path.replace(suffix, "sparql")
        sparql_path_list.append(output_path)
    destination_path = currentFolder + "/DWISVocabulary_sparql_patched.ttl"
    apply_sparql_to_KB(base_ttl_path, sparql_path_list, destination_path)


def delete_tmp_files(tmp_file_list: list):
    for file in tmp_file_list:
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"The file does not exist: {file}")


if __name__ == "__main__":
    currentFolder = os.path.dirname(os.path.realpath(__file__))

    apply_knowledge_patches_to_KB()

    # input_ttl_files_path = [
    #     currentFolder + "/DWISVocabulary_sparql_patched.ttl",
    #     # currentFolder + "/Units_Quantities.ttl",
    # ]
    # output_ttl_file_path = currentFolder + "/DWISVocabulary_merged_1.ttl"
    # merge_ttl_files(input_ttl_files_path, output_ttl_file_path)

    sparql_path_list = [currentFolder + "/patch_delete_update.sparql"]
    base_ttl_path = currentFolder + "/DWISVocabulary_sparql_patched.ttl"
    destination_path = currentFolder + "/DWISVocabulary_merged_patched.ttl"
    apply_sparql_to_KB(base_ttl_path, sparql_path_list, destination_path)

    input_ttl_files_path = [currentFolder + "/patch_create.ttl", currentFolder + "/DWISVocabulary_merged_patched.ttl"]
    output_ttl_file_path = currentFolder + "/DWISVocabulary_merged.ttl"
    merge_ttl_files(input_ttl_files_path, output_ttl_file_path)

    tmp_file_list = [
        currentFolder + "/DWISVocabulary_sparql_patched.ttl",
        # currentFolder + "/DWISVocabulary_merged_1.ttl",
        currentFolder + "/DWISVocabulary_merged_patched.ttl",
    ]
    delete_tmp_files(tmp_file_list)
