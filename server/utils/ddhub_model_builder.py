import json
import os

import yaml


def generate_Namespaces(project_name: str) -> str:
    sparql_str = r"""@prefix ddhub: <http://ddhub.no/>.
@prefix owl: <http://www.w3.org/2002/07/owl#>.
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>.
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.
@prefix : <Project_name>.
"""
    if project_name.endswith("/"):
        project_name = project_name[:-1] + "#"
    if not project_name.endswith("#"):
        project_name = project_name + "#"
    sparql_str = sparql_str.replace(r"Project_name", project_name)
    return sparql_str


def generate_DrillingDataPoint_PrototypeData(task: dict) -> str:
    """
    Input: DrillingDataPoint_name: str, PrototypeData_class: str
    """
    DrillingDataPoint_name = task["DrillingDataPoint_name"]
    PrototypeData_class = task["PrototypeData_class"]

    ddhub_model = r"""
:{myPoint} rdf:type owl:NamedIndividual;
    rdf:type ddhub:DrillingDataPoint;
    rdf:type ddhub:<PrototypeData>."""
    ddhub_model = ddhub_model.replace(r"{myPoint}", DrillingDataPoint_name)
    ddhub_model = ddhub_model.replace(r"<PrototypeData>", PrototypeData_class)
    return ddhub_model


def generate_MeasurableQuantity(task: dict) -> str:
    """
    Input: DrillingDataPoint_name: str, MeasurableQuantity_class: str
    """
    DrillingDataPoint_name = task["DrillingDataPoint_name"]
    MeasurableQuantity_class = task["MeasurableQuantity_class"]

    ddhub_model = r"""
:{DrillingDataPoint_name} ddhub:IsOfMeasurableQuantity ddhub:{MeasurableQuantity_name}."""
    ddhub_model = ddhub_model.replace(r"{DrillingDataPoint_name}", DrillingDataPoint_name)

    MeasurableQuantity_name = generate_standrard_instance_name(MeasurableQuantity_class, "Quantity")
    ddhub_model = ddhub_model.replace(r"{MeasurableQuantity_name}", MeasurableQuantity_name)
    return ddhub_model


def generate_standrard_instance_name(class_name: str, suffix_to_remove: str):
    standrard_instance_name = class_name[0].lower() + class_name[1:]
    standrard_instance_name = standrard_instance_name.replace(suffix_to_remove, "")
    return standrard_instance_name


def generate_Unit(task: dict) -> str:
    """
    Input: DrillingDataPoint_name: str, Unit_name: str, Unit_class: str, Quantity_class: str
    """
    DrillingDataPoint_name = task["DrillingDataPoint_name"]
    Unit_name = task["Unit_name"]
    Unit_class = task["Unit_class"]
    Quantity_class = task["Quantity_class"]

    ddhub_model = r"""
:{Unit_name} rdf:type owl:NamedIndividual;
    rdf:type ddhub:Unit;
    ddhub:IsUnitForQuantity ddhub:{Quantity_standardInstance}."""

    Quantity_standardInstance = generate_standrard_instance_name(Quantity_class, "Quantity")
    DrillingSignal_name = DrillingDataPoint_name + "_DrillingSignal"

    ddhub_model = ddhub_model.replace(r"{Unit_name}", Unit_name)
    ddhub_model = ddhub_model.replace(r"{Quantity_standardInstance}", Quantity_standardInstance)
    ddhub_model = ddhub_model.replace(r"{DrillingSignal_name}", DrillingSignal_name)

    if Unit_class != "None":
        ddhub_unit = r"""
:{Unit_name} rdf:type ddhub:{Unit_class};
    owl:sameAs ddhub:{Unit_standardInstance}."""
        Unit_standardInstance = generate_standrard_instance_name(Unit_class, "")
        ddhub_unit = ddhub_unit.replace(r"{Unit_name}", Unit_name)
        ddhub_unit = ddhub_unit.replace(r"{Unit_class}", Unit_class)
        ddhub_unit = ddhub_unit.replace(r"{Unit_standardInstance}", Unit_standardInstance)
        ddhub_model += ddhub_unit
    return ddhub_model


def generate_DynamicDrillingSignal(task: dict) -> str:
    """
    Input: DrillingDataPoint_name: str, Unit_name: str
    """
    DrillingDataPoint_name = task["DrillingDataPoint_name"]
    Unit_class = task["Unit_class"]
    if Unit_class == "None":
        Unit_name = ":" + task["Unit_name"]
    else:
        Unit_name = "ddhub:" + generate_standrard_instance_name(Unit_class, "")

    ddhub_model = r"""
:{DrillingSignal_name} rdf:type owl:NamedIndividual;
        rdf:type ddhub:DynamicDrillingSignal;
        ddhub:HasUnitOfMeasure {Unit_name}.
:{DrillingDataPoint_name} ddhub:HasDynamicValue :{DrillingSignal_name}."""
    DrillingSignal_name = DrillingDataPoint_name + "_DrillingSignal"
    ddhub_model = ddhub_model.replace(r"{DrillingSignal_name}", DrillingSignal_name)
    ddhub_model = ddhub_model.replace(r"{Unit_name}", Unit_name)
    ddhub_model = ddhub_model.replace(r"{DrillingDataPoint_name}", DrillingDataPoint_name)
    return ddhub_model


def generate_Provider(task: dict) -> str:
    """
    Input: DrillingDataPoint_name: str, Provider_name: str
    """
    DrillingDataPoint_name = task["DrillingDataPoint_name"]
    Provider_name = task["Provider_name"]

    template = r"""
:{myProvider} rdf:type owl:NamedIndividual;
        rdf:type ddhub:Provider.
:{myPoint} ddhub:IsProvidedBy :{myProvider}."""
    ddhub_model = ""
    if Provider_name != "":
        ddhub_model = template
        ddhub_model = ddhub_model.replace(r"{myPoint}", DrillingDataPoint_name)
        ddhub_model = ddhub_model.replace(r"{myProvider}", Provider_name)
    return ddhub_model


def generate_AllinOne(project_folder_path: str):
    with open(project_folder_path + "/task_batch.json", "r", encoding="utf-8") as file:
        task_batch = json.load(file)

    ttl_str = ""
    for task in task_batch.values():
        ttl_str += generate_DrillingDataPoint_PrototypeData(task)
        ttl_str += generate_MeasurableQuantity(task)
        ttl_str += generate_Unit(task)
        ttl_str += generate_DynamicDrillingSignal(task)
        ttl_str += generate_Provider(task) + "\n"
    ttl_str = generate_Namespaces(task["Namespace"]) + ttl_str
    with open(project_folder_path + "/ddhub_models.ttl", "w") as file:
        file.write(ttl_str)


if __name__ == "__main__":
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    generate_AllinOne(currentFolder + "/tasks/00001_short")
