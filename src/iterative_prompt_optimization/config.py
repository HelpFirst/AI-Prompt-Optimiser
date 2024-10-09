# Configuration settings for the iterative prompt optimization process

# Constants
MAX_RETRIES = 3  # Maximum number of retries for API calls

# Model selection options for different providers
MODEL_OPTIONS = {
    "ollama": ["llama2", "llama3.1", "mistral", "vicuna"],
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "azure_openai": ["gpt-35-turbo", "gpt-4"],
    "anthropic": ["claude-2", "claude-instant-1"],
    "google": ["gemini-pro"]
}

# Pricing per 1000 tokens
PRICING = {
    "azure_openai": {
        "gpt-35-turbo": 0.0008,
        "gpt-4": 0.06,
    },
    "openai": {
        "gpt-3.5-turbo": 0.0008,
        "gpt-4": 0.06,
    },
    "anthropic": {
        "claude-2": 0.01,
        "claude-instant-1": 0.0015,
    },
    "google": {
        "gemini-pro": 0.0005,  # Assuming similar to GPT-3.5
    }
}

# These will be set during runtime
SELECTED_PROVIDER = None
MODEL_NAME = None
EVAL_PROVIDER = None
EVAL_MODEL = None
EVAL_TEMPERATURE = 0.7
OPTIM_PROVIDER = None
OPTIM_MODEL = None
OPTIM_TEMPERATURE = 0.9

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys and configuration for different providers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_API_BASE')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
# AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_GPT35_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT35_DEPLOYMENT_NAME')
AZURE_OPENAI_GPT4_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT_NAME')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')


# Add this function to get the correct deployment name
def get_azure_deployment_name(model_name):
    if model_name == "gpt-35-turbo":
        return AZURE_OPENAI_GPT35_DEPLOYMENT_NAME
    elif model_name == "gpt-4":
        return AZURE_OPENAI_GPT4_DEPLOYMENT_NAME
    else:
        raise ValueError(f"Unknown Azure OpenAI model: {model_name}")

def set_model(provider: str, model: str) -> None:
    """
    Set the selected model provider and name.

    Args:
        provider (str): The selected AI provider
        model (str): The selected model name
    """
    global SELECTED_PROVIDER, MODEL_NAME
    SELECTED_PROVIDER = provider
    MODEL_NAME = model

def get_model() -> tuple:
    """
    Get the currently selected model provider and name.

    Returns:
        tuple: (SELECTED_PROVIDER, MODEL_NAME)
    """
    return SELECTED_PROVIDER, MODEL_NAME

def set_models(eval_provider: str, eval_model: str, eval_temperature: float, optim_provider: str, optim_model: str, optim_temperature: float) -> None:
    """
    Set the selected model providers, names, and temperatures for evaluation and optimization.

    Args:
        eval_provider (str): The selected AI provider for evaluation
        eval_model (str): The selected model name for evaluation
        eval_temperature (float): The temperature setting for evaluation
        optim_provider (str): The selected AI provider for optimization
        optim_model (str): The selected model name for optimization
        optim_temperature (float): The temperature setting for optimization
    """
    global EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMPERATURE, OPTIM_PROVIDER, OPTIM_MODEL, OPTIM_TEMPERATURE
    EVAL_PROVIDER = eval_provider
    EVAL_MODEL = eval_model
    EVAL_TEMPERATURE = eval_temperature
    OPTIM_PROVIDER = optim_provider
    OPTIM_MODEL = optim_model
    OPTIM_TEMPERATURE = optim_temperature

def get_eval_model() -> tuple:
    """
    Get the currently selected model provider, name, and temperature for evaluation.

    Returns:
        tuple: (EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMPERATURE)
    """
    return EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMPERATURE

def get_optim_model() -> tuple:
    """
    Get the currently selected model provider, name, and temperature for optimization.

    Returns:
        tuple: (OPTIM_PROVIDER, OPTIM_MODEL, OPTIM_TEMPERATURE)
    """
    return OPTIM_PROVIDER, OPTIM_MODEL, OPTIM_TEMPERATURE


# Import prompts
from .prompts import (
    PROMPT_ENGINEER_INPUT,
    FALSE_POSITIVES_ANALYSIS_PROMPT,
    FALSE_NEGATIVES_ANALYSIS_PROMPT,
    TRUE_POSITIVES_ANALYSIS_PROMPT,
    INVALID_OUTPUTS_ANALYSIS_PROMPT,
    VALIDATION_AND_IMPROVEMENT_PROMPT
)

# Example output schema structure
OUTPUT_SCHEMA_EXAMPLE = {
    'key_to_extract': 'risk_output',
    'value_mapping': {
        'risk_present': 1,
        'risk_not_present': 0
    },
    'regex_pattern': r'"risk_output":\s*"(.*?)"',
    'chain_of_thought_key': 'chain_of_thought',
    'chain_of_thought_regex': r'"chain_of_thought":\s*"(.*?)"'
}