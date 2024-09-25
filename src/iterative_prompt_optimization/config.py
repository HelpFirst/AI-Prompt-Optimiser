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

# Pricing per 1000 tokens (approximate as of 2023, may need updating)
PRICING = {
    "openai": {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.06,
    },
    "azure_openai": {
        "gpt-35-turbo": 0.002,
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
OPTIM_PROVIDER = None
OPTIM_MODEL = None

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys and configuration for different providers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_API_BASE')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')

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

def set_models(eval_provider: str, eval_model: str, optim_provider: str, optim_model: str) -> None:
    """
    Set the selected model providers and names for evaluation and optimization.

    Args:
        eval_provider (str): The selected AI provider for evaluation
        eval_model (str): The selected model name for evaluation
        optim_provider (str): The selected AI provider for optimization
        optim_model (str): The selected model name for optimization
    """
    global EVAL_PROVIDER, EVAL_MODEL, OPTIM_PROVIDER, OPTIM_MODEL
    EVAL_PROVIDER = eval_provider
    EVAL_MODEL = eval_model
    OPTIM_PROVIDER = optim_provider
    OPTIM_MODEL = optim_model

def get_eval_model() -> tuple:
    """
    Get the currently selected model provider and name for evaluation.

    Returns:
        tuple: (EVAL_PROVIDER, EVAL_MODEL)
    """
    return EVAL_PROVIDER, EVAL_MODEL

def get_optim_model() -> tuple:
    """
    Get the currently selected model provider and name for optimization.

    Returns:
        tuple: (OPTIM_PROVIDER, OPTIM_MODEL)
    """
    return OPTIM_PROVIDER, OPTIM_MODEL