"""
This module provides functionalities for interacting with a Large Language Model (LLM)
to obtain and process probability distributions over a set of predefined options.
It supports ranking options based on these probabilities, including aggregation
across multiple rounds of queries.
"""

import copy
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional, Union

sys.path.append(str(Path(__file__).resolve().parent))


from .. import chat_restapi as llm
from .. import utils as ut
from ..configs import log_config  # noqa: F401

logger = logging.getLogger(__name__)


def query_llm_for_probability_distribution(
    prompt_template: str,
    options: Union[list, tuple],
    options_placeholder: str,
    option_explanations: dict,
    option_explanations_placeholder: str,
    kvPairs: dict,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
) -> dict:
    """
    Queries the LLM to obtain a probability distribution over a given set of options.
    The LLM is configured to return log probabilities for a single token, which
    corresponds to the letter assigned to each option.

    Args:
        prompt_template (str): The template string for the prompt to be sent to the LLM.
        options (Union[list, tuple]): A list or tuple of candidate options.
        options_placeholder (str): The placeholder string in the prompt_template
                                   where the labeled options will be inserted.
        kvPairs (dict): A dictionary of key-value pairs to populate the prompt template.
        model (Optional[str]): The name of the LLM model to use. Defaults to llm.DEFAULT_MODEL.
        user_config (Optional[dict]): Additional configuration parameters for the LLM call.

    Returns:
        dict: A dictionary where keys are the original options and values are their
              corresponding probabilities as returned by the LLM.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    letter_map, reverse_map = ut.create_letter_mapping(options)
    options_labeled = [f"{letter}: {cand}" for cand, letter in letter_map.items()]
    # Prepare prompt
    kvPairs.update({options_placeholder: "\n".join(options_labeled)})
    kvPairs.update({option_explanations_placeholder: "\n".join(f"{key}: {value}" for key, value in option_explanations.items())})
    prompt = ut.assemble_prompt(prompt_template, kvPairs)
    logger.debug(f"Prompt: {prompt}")

    # Prepare LLM config
    config = user_config.copy() if user_config else {}
    config.update(
        {
            "logprobs": True,
            "temperature": 0.0,  # Use deterministic sampling
            "top_p": 1.0,  # Use full probability distribution
            "top_logprobs": min(20, len(letter_map)),
            "logit_bias": {ut.get_token_id(letter, model=model)[0]: 100 for letter in letter_map.values()},
            "max_completion_tokens": 1,
        }
    )

    # Execute the RAG task with the prepared prompt and configuration
    result = ut.run_rag_task_single(prompt, model=model, user_config=config)

    # Extract log probabilities from the LLM result
    # The 'logprobs' field is expected to be a list of token information.
    token_list = result.get("logprobs", [])
    # Assert that the LLM returned exactly one token, as configured.
    assert len(token_list) == 1, "LLM is not giving only one token."
    logger.debug(f"token_list: {token_list}")

    # Extract top log probabilities for the single token.
    # This is a list of dictionaries, each containing a token and its logprob.
    top_logprobs_for_single_token = token_list[0].get("top_logprobs", [])
    # Assert that the LLM considered all provided options.
    assert len(top_logprobs_for_single_token) == len(options), "LLM is not considering all the options."

    logger.debug(f"top_logprobs_for_single_token: {top_logprobs_for_single_token}")

    # Convert log probabilities to actual probabilities and map them back to original options.
    # Only include tokens that correspond to a valid option letter.
    probs_distribution = {reverse_map[item["token"]]: math.exp(item["logprob"]) for item in top_logprobs_for_single_token if item["token"] in reverse_map}
    logger.debug(f"probs_distribution result: {probs_distribution}")
    return probs_distribution


def _aggregate_probability_distributions(probs_distribution_list: list, sort_flag: bool = True) -> list:
    """
    Aggregates a list of probability distributions, calculating the average
    probability and retaining the history of probabilities for each option.

    Args:
        probs_distribution_list (list): A list of dictionaries, where each dictionary
                                        represents a probability distribution for a round.
                                        Example: [{"Option A": 0.8, "Option B": 0.1}, ...]

    Returns:
        list: A list of dictionaries, where each dictionary contains an option
              and its aggregated data (average probability and history).
              Example: [{"Option A": {"average": 0.805, "history": [0.8, 0.81]}}, ...]
    """
    # Step 1: Collect history for each option across all distributions
    history_dict = {}
    for entry in probs_distribution_list:
        for key, value in entry.items():
            if key not in history_dict:
                history_dict[key] = []
            history_dict[key].append(value)

    # Step 2: Compute average and format output
    # The output format is a list of single-item dictionaries for each option.
    probs_distribution_agg = [{key: {"average": sum(values) / len(values), "history": values}} for key, values in history_dict.items()]

    if sort_flag:
        # Sort the list of dictionaries by the 'average' value in descending order
        probs_distribution_agg.sort(key=lambda x: list(x.values())[0]["average"], reverse=True)
    return probs_distribution_agg


def rank_options_by_llm_probabilities(
    prompt_template: str,
    options: Union[list, tuple],
    options_placeholder: str,
    option_explanations: dict,
    option_explanations_placeholder: str,
    kvPairs: dict,
    rounds: int = 3,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
) -> list:
    """
    Ranks a list of options by querying the LLM multiple times (in 'rounds')
    to get probability distributions and then aggregating these distributions.
    In each round, options are shuffled to mitigate positional bias from the LLM.

    Args:
        prompt_template (str): The template string for the prompt.
        options (Union[list, tuple]): The list or tuple of options to be ranked.
        options_placeholder (str): The placeholder in the prompt_template for options.
        kvPairs (dict): Key-value pairs to fill the prompt template.
        rounds (int): The number of times to query the LLM and aggregate results.
                      Defaults to 5.
        model (Optional[str]): The LLM model to use. Defaults to llm.DEFAULT_MODEL.
        user_config (Optional[dict]): Additional configuration for the LLM call.

    Returns:
        list: An aggregated list of probability distributions for each option,
              including average probability and history across rounds.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    probs_distribution_list = []
    for round_num in range(rounds):
        # Create a deep copy of options and shuffle them to reduce positional bias
        shuffled_options = list(copy.deepcopy(options))
        random.shuffle(shuffled_options)

        # Get probability distribution for the current round
        probs_distribution = query_llm_for_probability_distribution(
            prompt_template=prompt_template,
            options=shuffled_options,
            options_placeholder=options_placeholder,
            option_explanations=option_explanations,
            option_explanations_placeholder=option_explanations_placeholder,
            kvPairs=kvPairs,
            model=model,
            user_config=user_config,
        )
        probs_distribution_list.append(probs_distribution)
        logger.info(f"Round {round_num + 1} probability distribution: {probs_distribution}")

    # Aggregate the probability distributions from all rounds
    probs_distribution_agg = _aggregate_probability_distributions(probs_distribution_list)
    return probs_distribution_agg


def rank_options_tournament(
    prompt_template: str,
    options: Union[list, tuple],
    options_placeholder: str,
    option_explanations: dict,
    option_explanations_placeholder: str,
    kvPairs: dict,
    rounds: int = 3,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
    pool_size: int = 10,
    cutoff_value: float = 0.2,
) -> list:
    """
    Ranks options using a tournament-style selection process.

    When the options size is greater than a threshold (pool_size), the options are
    separated into groups of max pool_size. The options with probability greater
    than a cutoff value will be winners and sent to a new option list.
    The selection process runs iteratively until a final set of options is determined.

    Args:
        prompt_template (str): The template string for the prompt.
        options (Union[list, tuple]): The list or tuple of options to be ranked.
        options_placeholder (str): The placeholder in the prompt_template for options.
        kvPairs (dict): Key-value pairs to fill the prompt template.
        rounds (int): The number of times to query the LLM and aggregate results
                      for each group in a tournament round. Defaults to 5.
        model (Optional[str]): The LLM model to use. Defaults to llm.DEFAULT_MODEL.
        user_config (Optional[dict]): Additional configuration for the LLM call.
        pool_size (int): The maximum number of options in a group for each tournament round.
                         Defaults to 10.
        cutoff_value (float): The probability threshold for an option to be considered
                              a "winner" and advance to the next round. Defaults to 0.2.

    Returns:
        list: A list of dictionaries, where each dictionary contains an option
              and its aggregated data (average probability and history) from the
              final ranking round.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    remaining_options = list(options)
    remaining_options_explanations = option_explanations
    tournament_round = 0

    while len(remaining_options) > pool_size:
        tournament_round += 1
        logger.info(f"Starting Tournament Round {tournament_round} with {len(remaining_options)} options.")
        next_round_options = []

        # Shuffle remaining options to ensure fair grouping
        random.shuffle(remaining_options)

        # Divide options into groups
        num_groups = math.ceil(len(remaining_options) / pool_size)

        for i in range(num_groups):
            start_index = i * pool_size
            end_index = min((i + 1) * pool_size, len(remaining_options))
            current_group = remaining_options[start_index:end_index]
            current_group_explanations = {k: option_explanations.get(k, {}) for k in current_group}
            logger.debug(f"Processing group {i + 1}/{num_groups} with {len(current_group)} options.")

            # Rank options within the current group
            ranked_group_options = rank_options_by_llm_probabilities(
                prompt_template=prompt_template,
                options=current_group,
                options_placeholder=options_placeholder,
                option_explanations=current_group_explanations,
                option_explanations_placeholder=option_explanations_placeholder,
                kvPairs=kvPairs,
                rounds=rounds,
                model=model,
                user_config=user_config,
            )

            # Identify winners based on cutoff_value
            group_winners = []
            for item in ranked_group_options:
                option_name = list(item.keys())[0]
                average_prob = item[option_name]["average"]
                if average_prob > cutoff_value:
                    group_winners.append(option_name)

            logger.debug(f"Group winners: {group_winners}")
            next_round_options.extend(group_winners)

        if not next_round_options:
            logger.info("No options passed the cutoff in this round. Breaking tournament.")
            break  # No options passed the cutoff, stop the tournament

        # Remove duplicates and update remaining options for the next round
        remaining_options = list(set(next_round_options))
        remaining_options_explanations = {k: option_explanations.get(k, {}) for k in remaining_options}
        logger.info(f"Tournament Round {tournament_round} completed. {len(remaining_options)} options advanced.")

    # Final ranking of the remaining options (could be one or more if the loop broke early)
    if remaining_options:
        logger.info(f"Performing final ranking on {len(remaining_options)} remaining options.")
        final_ranked_options = rank_options_by_llm_probabilities(
            prompt_template=prompt_template,
            options=remaining_options,
            options_placeholder=options_placeholder,
            option_explanations=remaining_options_explanations,
            option_explanations_placeholder=option_explanations_placeholder,
            kvPairs=kvPairs,
            rounds=rounds,
            model=model,
            user_config=user_config,
        )
        return final_ranked_options
    else:
        logger.info("No options remained after the tournament process.")
        return []
