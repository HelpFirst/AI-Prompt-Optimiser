# Configuration module for the iterative prompt optimization process
# This module manages model selection, API configurations, and pricing settings

# Maximum number of retries for API calls before falling back to default behavior
MAX_RETRIES = 3

# Available model options for each provider
# Maps provider names to their supported models for easy selection
MODEL_OPTIONS = {
    "ollama": ["llama2", "llama3.1", "mistral", "vicuna"],
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "azure_openai": ["gpt-35-turbo", "gpt-4"],
    "anthropic": ["claude-2", "claude-instant-1"],
    "google": ["gemini-pro"]
}

# Cost per 1000 tokens for each model by provider
# Used for estimating API costs before running optimizations
PRICING = {
    "azure_openai": {
        "gpt-35-turbo": 0.0008,
        "gpt-4": 0.06,
    },
    "openai": {
        "gpt-3.5-turbo": 0.0008,
        "gpt-4": 0.06,
        "gpt-4o-mini": 0.0012,
    },
    "anthropic": {
        "claude-2": 0.01,
        "claude-instant-1": 0.0015,
    },
    "google": {
        "gemini-pro": 0.0005,  # Similar pricing to GPT-3.5
    }
}

# Global configuration variables
# These will be set during runtime based on user selection
SELECTED_PROVIDER = None  # Current AI provider
MODEL_NAME = None        # Current model name
EVAL_PROVIDER = None     # Provider for evaluation
EVAL_MODEL = None       # Model for evaluation
EVAL_TEMPERATURE = 0.7  # Temperature setting for evaluation
OPTIM_PROVIDER = None   # Provider for optimization
OPTIM_MODEL = None      # Model for optimization
OPTIM_TEMPERATURE = 0.9 # Temperature setting for optimization

# Load environment variables and API configurations
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# API keys and configuration settings for different providers
# These are loaded from environment variables for security
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_API_BASE')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_GPT35_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT35_DEPLOYMENT_NAME')
AZURE_OPENAI_GPT4_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT_NAME')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')  # Default to llama3.1
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')  # Default local endpoint

def get_azure_deployment_name(model_name: str) -> str:
    """
    Get the correct Azure deployment name based on the model.
    
    Args:
        model_name (str): Name of the Azure OpenAI model
        
    Returns:
        str: Corresponding deployment name
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "gpt-35-turbo":
        return AZURE_OPENAI_GPT35_DEPLOYMENT_NAME
    elif model_name == "gpt-4":
        return AZURE_OPENAI_GPT4_DEPLOYMENT_NAME
    else:
        raise ValueError(f"Unknown Azure OpenAI model: {model_name}")

# Functions to manage model selection and configuration
def set_model(provider: str, model: str) -> None:
    """Set the global model provider and name."""
    global SELECTED_PROVIDER, MODEL_NAME
    SELECTED_PROVIDER = provider
    MODEL_NAME = model

def get_model() -> tuple:
    """Get the current model provider and name."""
    return SELECTED_PROVIDER, MODEL_NAME

def set_models(eval_provider: str, eval_model: str, eval_temperature: float, 
              optim_provider: str, optim_model: str, optim_temperature: float) -> None:
    """
    Configure models for both evaluation and optimization phases.
    Sets global variables for providers, models, and their temperatures.
    """
    global EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMPERATURE, OPTIM_PROVIDER, OPTIM_MODEL, OPTIM_TEMPERATURE
    EVAL_PROVIDER = eval_provider
    EVAL_MODEL = eval_model
    EVAL_TEMPERATURE = eval_temperature
    OPTIM_PROVIDER = optim_provider
    OPTIM_MODEL = optim_model
    OPTIM_TEMPERATURE = optim_temperature

def get_eval_model() -> tuple:
    """Get the current evaluation model configuration."""
    return EVAL_PROVIDER, EVAL_MODEL, EVAL_TEMPERATURE

def get_optim_model() -> tuple:
    """Get the current optimization model configuration."""
    return OPTIM_PROVIDER, OPTIM_MODEL, OPTIM_TEMPERATURE

# Example schema for output formatting
# This defines how model outputs should be structured and parsed
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
