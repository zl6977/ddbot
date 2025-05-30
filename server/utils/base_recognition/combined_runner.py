import logging

from .. import chat_restapi as llm
from ..configs import globals_config as glb
from . import recognizer_pc as rec_pc
from . import recognizer_pc_cot as rec_pc_cot

logger = logging.getLogger(__name__)

def interpret_mnemonic(
    user_query: str,
    CoT_flag: bool,
    model: str = llm.DEFAULT_MODEL,
):
    if CoT_flag:
        prompt_templates = glb.prompt_template_collection["PC_CoT"]
        interpretation, prompt = rec_pc_cot.interpret_mnemonic(
            user_query, glb.complementary_knowledge, prompt_templates, model
        )
    else:
        prompt_templates = glb.prompt_template_collection["PC"]
        interpretation, prompt = rec_pc.interpret_mnemonic(
            user_query, glb.complementary_knowledge, prompt_templates, model
        )
    return interpretation, prompt


def recognize_metadata(task_key: str, task_data: dict, models_high_low: list = [llm.DEFAULT_MODEL, llm.DEFAULT_MODEL]) -> tuple:
    """Baseline recognition only; returns raw recognition_output for post-processing."""

    user_query = task_data.get("Raw_content", "")
    logger.info(f"[base] Running task for user query: {user_query}")
    interpretation = task_data.get("Interpretation_user", "")
    logger.info(f"[base] Task interpretation: {interpretation}")

    task_control = task_data.get("TaskControl", {})
    CoT_flag = task_control.get("Chain_of_Thought", False)
    number_of_candidates = task_control.get(
        "number_of_candidates",
        {"Quantity_class": 5, "Unit_class": 10, "PrototypeData_class": 5},
    )
    # Validation controls are consumed inside post-recognition tasks

    if CoT_flag:
        prompt_templates = glb.prompt_template_collection["PC_CoT"]
        recognition_output = rec_pc_cot.recoginize_metadata(
            user_query,
            interpretation,
            glb.complementary_knowledge,
            prompt_templates,
            glb.quantity_fullList_extraContent,
            glb.unit_fullList_extraContent,
            glb.prototypeData_fullList_extraContent,
            models_high_low,
            number_of_candidates,
        )
    else:
        prompt_templates = glb.prompt_template_collection["PC"]
        recognition_output = rec_pc.recoginize_metadata(
            user_query,
            interpretation,
            glb.complementary_knowledge,
            prompt_templates,
            glb.quantity_fullList_extraContent,
            glb.unit_fullList_extraContent,
            glb.prototypeData_fullList_extraContent,
            models_high_low,
            number_of_candidates,
        )

    return recognition_output