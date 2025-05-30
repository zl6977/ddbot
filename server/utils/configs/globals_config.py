import json
import logging
import os
import time
from datetime import datetime

import yaml

from .. import sparql_connector as sc

logger = logging.getLogger(__name__)

server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ui_dir = os.path.join(server_dir, os.pardir, "ui")
tasks_dir = os.path.join(server_dir, "tasks")
tmp_dir = os.path.join(server_dir, "tmp")
default_log_path = os.path.join(tmp_dir, f"server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

utils_dir = os.path.abspath(os.path.dirname(__file__))
prompt_template_path = {
        "Generic": os.path.join(utils_dir, os.pardir, "prompt_resources", "templates_generic.yaml"),
        "PC": os.path.join(utils_dir, os.pardir, "prompt_resources", "templates_pc.yaml"),
        "PC_CoT": os.path.join(utils_dir, os.pardir, "prompt_resources", "templates_pc_cot.yaml"),
}
complementary_knowledge_path = os.path.join(utils_dir, os.pardir, os.pardir, os.pardir, "data_store", "complementary_knowledge.yaml")


prompt_template_collection = {}
complementary_knowledge = {}
quantity_fullList_extraContent = {}
unit_fullList_extraContent = {}
prototypeData_fullList_extraContent = {}


def load_files():
    global prompt_template_collection, complementary_knowledge, quantity_fullList_extraContent, unit_fullList_extraContent, prototypeData_fullList_extraContent
    
    for k, v in prompt_template_path.items():
        with open(v, "r") as file:
            pt = yaml.safe_load(file)
        prompt_template_collection.update({k: pt})

    with open(complementary_knowledge_path, "r") as file:
        complementary_knowledge.update(yaml.safe_load(file))

    logger.info("Start retrieving knowledge.")
    quantity_fullList_extraContent = sc.generate_Quantity_fullList_extraContent(sc.KB_ttl_path)
    unit_fullList_extraContent = sc.generate_Unit_fullList_extraContent(sc.KB_ttl_path)
    prototypeData_fullList_extraContent = sc.generate_PrototypeData_fullList_extraContent(sc.KB_ttl_path)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with open(tmp_dir + "/quantity_fullList_extraContent.json", "w") as json_file:
        json.dump(quantity_fullList_extraContent, json_file, indent=4)
    with open(tmp_dir + "/unit_fullList_extraContent.json", "w") as json_file:
        json.dump(unit_fullList_extraContent, json_file, indent=4)
    with open(tmp_dir + "/prototypeData_fullList_extraContent.json", "w") as json_file:
        json.dump(prototypeData_fullList_extraContent, json_file, indent=4)
    logger.info("Finish retrieving knowledge.")



start_time = time.time()
load_files()
end_time = time.time()
logger.info(f"Knowledge retrieval completed in {end_time - start_time} seconds.")