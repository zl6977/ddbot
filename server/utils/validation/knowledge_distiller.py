import json
import logging
import os
from typing import Dict, Optional, Tuple

import yaml

from .. import chat_restapi as llm
from .. import utils as ut
from ..configs import globals_config as glb

logger = logging.getLogger(__name__)


def distill_knowledge(
    user_query: str,
    interpretation: str,
    recognized_class: dict,
    model: str = llm.DEFAULT_MODEL,
) -> Tuple[bool, Optional[str]]:
    """
    Distill new knowledge from successful recognition results.
    
    Args:
        user_query: The original raw metadata query
        interpretation: User interpretation of the mnemonic
        complementary_knowledge: Current complementary knowledge base
        recognized_class: The successfully recognized classes from validation
        prompt_templates: Dictionary containing prompt templates
        model: LLM model to use for distillation
    
    Returns:
        Tuple of (has_new_insight: bool, knowledge_entry: Optional[str])
        - has_new_insight: True if new knowledge was gained, False otherwise
        - knowledge_entry: New knowledge entry in format "{mnemonic}: {description}" or None
    """
    logger.debug("Starting knowledge distillation")

    distill_prompt_set = glb.prompt_template_collection['Generic']['KnowledgeDistillation']

    # Prepare the prompt using standard assemble_prompt function
    prompt_template = distill_prompt_set.get("distillation", "")
    kvPairs = distill_prompt_set.get("placeholders", [])
    kvPairs = {
        k: "" for k in kvPairs
    }

    # Format complementary knowledge for prompt
    complementary_knowledge_str = glb.complementary_knowledge.get("basic", "")
    
    # Format recognized classes for prompt
    recognized_class_str = json.dumps(recognized_class, indent=2)
    
    # Prepare key-value pairs for standard prompt assembly
    kvPairs.update({
        "<user_query>": user_query,
        "<interpretation>": interpretation or "",
        "<complementary_knowledge>": complementary_knowledge_str,
        "<recognized_class>": recognized_class_str,
    })
    
    # Assemble the prompt using the standard function
    prompt = ut.assemble_prompt(prompt_template, kvPairs)
    
    try:
        # Call LLM for knowledge distillation
        logger.debug(f"Calling LLM for knowledge distillation with model: {model}")
        response = llm.chat_with_llm(prompt, model=model)
        
        if not response:
            logger.warning("Empty response from LLM during knowledge distillation")
            return False, None
        
        # Extract the content from the response
        result = llm.result_extractor(response, model)
        response_text = result.get("content", "")
        
        if not response_text:
            logger.warning("No content in LLM response during knowledge distillation")
            return False, None
            
        logger.debug(f"Knowledge distillation response: {response_text}")
        
        # Check if no new insight was gained
        if "No new insight" in response_text or "no new insight" in response_text.lower():
            logger.info("No new insight gained from recognition")
            return False, None
        
        # Extract knowledge entry (expecting format: "MNEMONIC: description")
        knowledge_entry = _extract_knowledge_entry(response_text)
        
        if knowledge_entry:
            logger.info(f"New knowledge distilled: {knowledge_entry}")
            return True, knowledge_entry
        else:
            logger.warning("Could not extract valid knowledge entry from response")
            return False, None
            
    except Exception as e:
        logger.error(f"Error during knowledge distillation: {e}")
        return False, None


def _extract_knowledge_entry(response_text: str) -> Optional[str]:
    """
    Extract knowledge entry from LLM response.
    
    Args:
        response_text: Raw response from LLM
    
    Returns:
        Formatted knowledge entry string or None if extraction fails
    """
    # Clean up the response
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    for line in lines:
        # Look for pattern "MNEMONIC: description"
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                mnemonic_part = parts[0].strip()
                description_part = parts[1].strip()
                
                # Basic validation - check that we have meaningful content
                if (description_part and 
                    len(description_part) > 3 and 
                    mnemonic_part and 
                    len(mnemonic_part) > 0):
                    
                    # Format the knowledge entry
                    knowledge_entry = f"{mnemonic_part}: {description_part}"
                    return knowledge_entry
    
    return None

def str_presenter(dumper, data):
    if '\n' in data:  # only use '|' style for multiline strings
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def save_knowledge_to_complementary_knowledge(knowledge_entry: str, complementary_knowledge_path: Optional[str] = None) -> bool:
    """
    Save a new knowledge entry to the complementary_knowledge.yaml file.
    
    Args:
        knowledge_entry: Knowledge entry in format "MNEMONIC: description"
        complementary_knowledge_path: Path to complementary_knowledge.yaml (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine the path to complementary_knowledge.yaml
        if complementary_knowledge_path is None:
            complementary_knowledge_path = glb.complementary_knowledge_path
        # Read the current complementary knowledge
        with open(complementary_knowledge_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        
        if data is None:
            data = {}
        
        # Ensure the basic section exists
        if "basic" not in data:
            data["basic"] = ""
        
        # Format the knowledge entry
        
        # Add the new knowledge entry to the basic section
        current_basic = data["basic"].strip()
        if current_basic:
            if not current_basic.endswith('\n'):
                current_basic += '\n'
            data["basic"] = current_basic + knowledge_entry
        else:
            data["basic"] = knowledge_entry

        data['basic'] = "\n".join(
            line.strip() for line in data['basic'].splitlines() if line.strip()
        )
        
        yaml.add_representer(str, str_presenter)
        

        # Write back to the file
        with open(complementary_knowledge_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Successfully saved knowledge entry to complementary knowledge: {knowledge_entry}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving knowledge entry to complementary knowledge: {e}")
        return False


def update_complementary_knowledge_in_memory(knowledge_entry: str) -> bool:
    """
    Update the in-memory complementary_knowledge dictionary with a new entry.
    
    Args:
        knowledge_entry: Knowledge entry in format "MNEMONIC: description"
        complementary_knowledge: The in-memory complementary knowledge dictionary
    
    Returns:
        True if successful, False otherwise
    """
    try:
         
        # Add the new knowledge entry to the basic section
        current_basic = glb.complementary_knowledge.get("basic", "").strip()
        if current_basic:
            if not current_basic.endswith('\n'):
                current_basic += '\n'
            glb.complementary_knowledge["basic"] = current_basic + knowledge_entry
        else:
            glb.complementary_knowledge["basic"] = knowledge_entry

        logger.debug(f"Updated in-memory complementary knowledge with: {knowledge_entry}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating in-memory complementary knowledge: {e}")
        return False
