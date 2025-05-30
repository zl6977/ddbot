import json
import logging
import os
import time
from copy import deepcopy
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# DEFAULT_MODEL = "gpt-5-nano"
# DEFAULT_MODEL = "gpt-4.1-mini"
# DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"
DEFAULT_MODEL = "gpt-4o-mini"
HIGH_MODEL = "gpt-5-nano"
LOW_MODEL = "gpt-4.1-mini"

LLM_MODELS = {
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-5-nano",
        # "gpt-4o-mini-2024-07-18",
        # "gpt-4o",
        # "gpt-4o-2024-08-06",
        # "gpt-4.1-nano",
        # "gpt-4.1-nano-2025-04-14",
        # "gpt-4.1-mini",
        # "gpt-4.1-mini-2025-04-14",
        # "gpt-4.1",
        # "gpt-4.1-2025-04-14",
    ],
    "OpenAI_reasoning": ["o3-mini", "o3-mini-2025-01-31"],
    "Ollama": ["llama3.1", "qwen2", "deepseek-r1"],
    "SiliconFlow": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"],
    "OpenRouter": ["deepseek/deepseek-r1:free", "deepseek/deepseek-r1"],
    "Gemini": [
        "gemini-2.5-flash-preview-05-20",
        "gemini-pro",
    ],
}


def find_service_provider(model_name: str) -> str:
    for provider, models in LLM_MODELS.items():
        if model_name in models:
            return provider
    return "Unknown provider"


def request_with_retry(
    url: str,
    headers: dict,
    data: dict,
    max_retries=5,
    initial_retry_delay=2,
    timeout=(3, 600),
):
    retry_delay = initial_retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            if 200 <= response.status_code < 300:
                return response

            # https://help.openai.com/en/articles/6891839-api-error-codes
            # Retry can not help: 401, 403, 404
            if response.status_code in {401, 403, 404}:
                logger.error(f"Try {attempt}/{max_retries}: Received {response.status_code} error. Check the server.")
                response.raise_for_status()

            # For error 429, retry_delay = Retry-After in header
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_time = int(retry_after) if retry_after is not None else retry_delay
                except ValueError:
                    wait_time = retry_delay
                logger.error(f"Try {attempt}/{max_retries}: Received 429 error. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 400:
                logger.debug(f"Try {attempt}/{max_retries}: Received 400 error. Check the request data.")
                response.raise_for_status()
            else:
                # Retry for other cases: e.g., 500, 502, 503, 504
                logger.error(f"Try {attempt}/{max_retries}: Received {response.status_code} error. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)

            # exponential backoff
            retry_delay *= 2

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(f"Try {attempt}/{max_retries}: {type(e).__name__} occurred. Retrying after {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except requests.RequestException as e:
            logger.error(f"Try {attempt}/{max_retries}: RequestException occurred: {e}")
            raise

    raise Exception("Max retries exceeded")


def result_extractor(response: Optional[requests.Response], model: str) -> dict:
    if not isinstance(response, requests.Response):
        return {}
    dict_return = {
        "status_code": response.status_code,
        "status_text": response.text,
    }

    try:
        response_data = response.json()
    except json.JSONDecodeError:
        logger.error("Invalid JSON: ", response.text)
        return dict_return

    dict_return.update(
        {
            "model": response_data["model"],
            "content": ["To be updated"],
        }
    )

    service_provider = find_service_provider(model)
    extractor = SERVICE_PROVIDERS[service_provider]["result_extractor"]
    try:
        response_content = extractor(response_data)
        dict_return.update(response_content)
    except Exception as e:
        logger.error("Can not extract from response_data: " + str(response_data))
        logger.exception("Exception occurred: %s", e)
    return dict_return


def result_extractor_openai(response_data: dict) -> dict:
    dict_return = {}
    # result_list = response_data["choices"][0]["message"]["content"].split(",")
    # dict_return["content"] = [s.strip() for s in result_list]
    dict_return["content"] = response_data["choices"][0]["message"]["content"]
    dict_return["prompt_tokens"] = response_data["usage"]["prompt_tokens"]
    dict_return["completion_tokens"] = response_data["usage"]["completion_tokens"]
    logprobs = response_data["choices"][0].get("logprobs")
    dict_return["logprobs"] = logprobs["content"] if logprobs and "content" in logprobs else None
    return dict_return


def result_extractor_ollama(response_data: dict) -> dict:
    dict_return = {}
    result_content = response_data["message"]["content"]
    if response_data["model"] == "deepseek-r1":
        thinking_answer = separate_content_in_labels(result_content, "<think>")
        result_content = thinking_answer[1]
        result_reasoning_content = thinking_answer[0]
        dict_return["reasoning_content"] = result_reasoning_content
    # result_list = result_content.split(",")
    # dict_return["content"] = [s.strip() for s in result_list]
    dict_return["content"] = result_content
    return dict_return


def result_extractor_siliconflow(response_data: dict) -> dict:
    dict_return = {}
    result_content = response_data["choices"][0]["message"]["content"]
    if "reasoning_content" in response_data["choices"][0]["message"]:
        result_reasoning_content = response_data["choices"][0]["message"]["reasoning_content"]
        dict_return["reasoning_content"] = result_reasoning_content
    # result_list = result_content.split(",")
    # dict_return["content"] = [s.strip() for s in result_list]
    dict_return["content"] = result_content
    return dict_return


def result_extractor_openrouter(response_data: dict) -> dict:
    dict_return = {}
    result_content = response_data["choices"][0]["message"]["content"]
    if "reasoning" in response_data["choices"][0]["message"]:
        result_reasoning_content = response_data["choices"][0]["message"]["reasoning"]
        dict_return["reasoning_content"] = result_reasoning_content
    # result_list = result_content.split(",")
    # dict_return["content"] = [s.strip() for s in result_list]
    dict_return["content"] = result_content
    return dict_return


def result_extractor_gemini(response_data: dict) -> dict:
    dict_return = {}
    try:
        dict_return["content"] = response_data.get("choices", [{}])[0].get("message", {}).get("content")
        usage = response_data.get("usage", {})
        dict_return["prompt_tokens"] = usage.get("prompt_tokens")
        dict_return["completion_tokens"] = usage.get("completion_tokens")
        dict_return["logprobs"] = response_data.get("choices", [{}])[0].get("logprobs", {}).get("content")
    except Exception:
        logger.exception("Failed to extract fields from response_data: %s", response_data)
    return dict_return


def separate_content_in_labels(content: str, label: str = "<think>") -> list:
    end_label = f"</{label[1:]}"
    content_splitted = content.split(end_label)
    content_splitted[0] = content_splitted[0].replace(label, "")
    return content_splitted


def chat_with_llm(prompt: str, system_prompt: str = "", model: str = DEFAULT_MODEL, user_config: Optional[dict] = None):
    response = {}
    service_provider = find_service_provider(model)

    if service_provider in SERVICE_PROVIDERS.keys():
        provider_config = deepcopy(SERVICE_PROVIDERS[service_provider])
        if user_config is not None:
            provider_config["data_config"].update(user_config)

        headers = {"Content-Type": "application/json"}
        if provider_config["API_KEY"] is not None:
            headers["Authorization"] = f"Bearer {provider_config['API_KEY']}"

        message = [{"role": "system", "content": system_prompt}] if system_prompt else []

        data = {
            "model": model,
            "messages":  message + [{"role": "user", "content": f"{prompt}"}],
            "stream": False,
        }
        data.update(provider_config["data_config"])

        response = request_with_retry(provider_config["API_URL"], headers=headers, data=data, max_retries=8)
        return response
    else:
        logger.error(f"Service provider {service_provider} is not supported.")


# Main function to interact with the user
def chat_multiround():
    print("Chat with LLM! Type 'exit' to end the chat.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
            break
        # response = chat_with_ollama(user_message)
        # print(f"Ollama: {response}")

        model = "gpt-4o-mini"
        response = chat_with_llm(user_message, model=model)
        result = result_extractor(response, model=model)
        print(f"LLM: {result}")


SERVICE_PROVIDERS = {
    "OpenAI": {
        "API_URL": "https://api.openai.com/v1/chat/completions",
        "API_KEY": os.getenv("OPENAI_API_KEY"),
        "data_config": {
            "temperature": 0.1,
            # "logprobs": True,
            # "top_logprobs": 5,
        },
        "result_extractor": result_extractor_openai,
    },
    "OpenAI_reasoning": {
        "API_URL": "https://api.openai.com/v1/chat/completions",
        "API_KEY": os.getenv("OPENAI_API_KEY"),
        "data_config": {},
        "result_extractor": result_extractor_openai,
    },
    "Ollama": {
        "API_URL": "http://localhost:11434/api/chat",
        "API_KEY": None,
        "data_config": {
            "temperature": 0.1,
        },
        "result_extractor": result_extractor_ollama,
    },
    "SiliconFlow": {
        "API_URL": "https://api.siliconflow.com/v1/chat/completions",
        "API_KEY": os.getenv("SiliconFlow_API_KEY"),
        "data_config": {
            "temperature": 0.1,
        },
        "result_extractor": result_extractor_siliconflow,
    },
    "OpenRouter": {
        "API_URL": "https://openrouter.ai/api/v1/chat/completions",
        "API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "data_config": {
            "temperature": 0.1,
        },
        "result_extractor": result_extractor_openrouter,
    },
    "Gemini": {
        "API_URL": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "API_KEY": os.getenv("GEMINI_API_KEY"),
        "data_config": {
            "temperature": 0.1,
        },
        "result_extractor": result_extractor_gemini,
    },
}


def test_chat_api():
    prompt = "Instructions: You will receive a user input containing mnemonic-based metadata used in the drilling domain. Preselect the quantities related to the metadata from the provided list. An interpretation will be provided for your reference, but its correctness is not guaranteed. Some complementary knowledge are {{'SPP': 'Stand Pipe Pressure.', 'SPM': 'Strokes Per Minutes, SPM is a type of PumpRate.', 'RPM': 'Revolutions Per Minute. RPM means SurfaceRPM not DownholeRPM unless specified.', 'ROP': 'Rate of Penetration.', 'GS': 'GeoService.', 'SRV': 'Survey.', 'HKLD': 'Hookload.', 'HKD': 'Hookload.', 'POS': 'Position.', 'WOB': 'Weight of Bit.', 'SWOB': 'Surface Weight of Bit.', 'TV': 'Tank volume.', 'TQ': 'Torque.', 'WT': 'Weight.'}}. Input: User input: {{'Mnemonic': 'SWOB30s', 'Description': 'zzz:undefined', 'DataType': 'double', 'Unit': 'kkgf'}}. Interpretation: {['{SWOB: Surface Weight of Bit', '30s: average over 30 seconds}']}. Quantity list: {['AccelerationQuantity', 'AmountSubstanceQuantity', 'AngleGradientPerLengthQuantity', 'AngleMagneticFluxDensityQuantity', 'AngularAccelerationQuantity', 'AngularVelocityQuantity', 'AreaQuantity', 'CompressibilityQuantity', 'CurvatureQuantity', 'DimensionlessQuantity', 'DynamicViscosityQuantity', 'ElectricCapacitanceQuantity', 'ElectricCurrentQuantity', 'ElectricResistivityQuantity', 'ElongationGradientPerLengthQuantity', 'EnergyDensityQuantity', 'EnergyQuantity', 'ForceGradientPerLengthQuantity', 'ForceQuantity', 'FrequencyQuantity', 'FrequencyRateOfChangeQuantity', 'GravitationalLoadQuantity', 'HeatTransferCoefficientQuantity', 'HydraulicConductivityQuantity', 'InterfacialTensionQuantity', 'IsobaricSpecificHeatCapacityGradientPerTemperatureQuantity', 'IsobaricSpecificHeatCapacityQuantity', 'LengthQuantity', 'LuminousIntensityQuantity', 'MagneticFluxDensityQuantity', 'MagneticFluxQuantity', 'MassDensityGradientPerLengthQuantity', 'MassDensityGradientPerTemperatureQuantity', 'MassDensityQuantity', 'MassDensityRateOfChangeQuantity', 'MassGradientPerLengthQuantity', 'MassQuantity', 'MassRateQuantity', 'MaterialStrengthQuantity', 'PlaneAngleQuantity', 'PorousMediumPermeabilityQuantity', 'PowerQuantity', 'PressureGradientPerLengthQuantity', 'PressureLossConstantQuantity', 'PressureQuantity', 'ProportionQuantity', 'RandomWalkQuantity', 'RelativeTemperatureQuantity', 'RotationalFrequencyRateOfChangeQuantity', 'SolidAngleQuantity', 'StressQuantity', 'TemperatureGradientPerLengthQuantity', 'TemperatureQuantity', 'TensionQuantity', 'ThermalConductivityGradientPerTemperatureQuantity', 'ThermalConductivityQuantity', 'TimeQuantity', 'TorqueGradientPerLengthQuantity', 'TorqueQuantity', 'VelocityQuantity', 'VolumeQuantity', 'VolumetricFlowRateOfChangeQuantity', 'VolumetricFlowRateQuantity', 'WaveNumberQuantity']}. Output: Return the top 3 selected quantities only, formatted as 'item1, item2, item3', without any explanation or additional information. If no match or less than 3 matches are found, use 'None' as a placeholder."
    # response = chat_with_ollama(prompt, "deepseek-r1")
    # data_config = {"logit_bias": {8459: -100, 10843: -100, 6: 50, 27384: 50, 21757: 50, 1366: 50, 8505: 50}}
    data_config = {
        # "model": "deepseek/deepseek-r1:free",
        # "reasoning_effort": "medium",
        "temperature": 0.1,
    }
    # response = chat_with_ds_like(prompt, model="deepseek/deepseek-r1:free", data_config=data_config)
    # result = result_extractor(response)
    model = "deepseek/deepseek-r1:free"
    model = "deepseek-ai/DeepSeek-V3"
    model = "gpt-4.1-nano"
    model = "gemini-2.5-flash-preview-05-20"
    # model = "gpt-4o-mini"
    response = chat_with_llm(prompt, model=model, user_config=data_config)
    result = result_extractor(response, model=model)
    # result_list = response_data["message"]["content"].split(",")
    print("result content: ", result["content"])


if __name__ == "__main__":
    test_chat_api()
