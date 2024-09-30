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
# AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_GPT35_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT35_DEPLOYMENT_NAME')
AZURE_OPENAI_GPT4_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT_NAME')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
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


ANALYSIS_PROMPT = """
You are an expert in refining LLMs prompts used in classification tasks. You will be analyzing misclassifications for the following base prompt:

Base Prompt:
{initial_prompt}

Below are two sets of texts that were misclassified by the LLM model.
You need to think and answer the questions of: why was this text misclassified? What led to be incorrectly inferred? What could be some instructions/rules to prevent this from been misclassified?

    Negative (0) texts (incorrectly classified as positive):
    {fp_texts}

    Positives (1) texts (incorrectly classified as negative):
    {fn_texts}

Your task is to analyze these misclassifications, with a special focus on improving recall while maintaining good precision. Pay particular attention to the false negatives (texts incorrectly classified as negative), and identify specific examples where the model missed subtle indicators of positive cases.

Based on your analysis, suggest strategies to improve the classification prompt - and how to craft it - focusing on how it can better recognize positive cases that were missed, and false warnings (texts incorrectly classified as positive). Your recommendations should include ways to reduce true and false negatives. Try to analyze the instances where the LLM may have inferred a risk present when it was not explicitly stated - and potentially improve the prompt to avoid this false positives.

-----

If there were invalid outputs, also suggest ways to improve the prompt to ensure the model follows the output format instructions more consistently. If 'Invalid output messages:' is empty, please ignore this subtask.

Invalid output messages:
{invalid_output_message}

Expected output format:
{output_format}

Note: The output should strictly adhere to the format specified above. Any deviation from this format will be considered an invalid output.
"""

PROMPT_ENGINEER_INPUT = """
You are an expert in crafting highly effective prompts. Your task is to help me improve a given prompt, with a specific focus on reducing false and false positives. I will give you the current prompt, output format, an analysis showing where it failed to classify a piece of text correctly (especially false negatives), and the metrics from the previous iteration. Your goal is to refine the prompt to be more precise and adaptable, ensuring that the AI can accurately classify similar texts going forward, with a particular emphasis on correctly identifying positive cases.

Current prompt:
{initial_prompt}

Analysis of misclassifications:
{analysis}

Metrics from the previous iteration:
- Precision: {precision}
- Recall: {recall}
- Accuracy: {accuracy}
- F1: {f1}

If precision is low, please focus on reducing false positives. If recall is low, please focus on strategies for reducing false negatives.

Please provide an improved prompt that addresses the issues identified in the analysis. The revised prompt should be written in the first person, guiding the AI to handle difficult or edge cases. Ensure that the new prompt is compatible with the given output format. Please output the final 'production-ready' prompt only, with no headers or other text.
"""

# Analysis prompts
FALSE_POSITIVES_ANALYSIS_PROMPT = """
Analyze the following false positive examples for the given prompt:

Current Prompt:
{initial_prompt}

False Positive Examples:
{fp_texts}

Your task is to analyze these false positives and suggest improvements to the prompt that would increase precision and reduce false warnings. Focus on:
1. Common patterns or characteristics in these false positives
2. Specific elements of the current prompt that might be causing these misclassifications
3. Concrete suggestions to modify the prompt to reduce false positives while maintaining overall accuracy

Provide your analysis and suggestions in a clear, structured format.
"""

FALSE_NEGATIVES_ANALYSIS_PROMPT = """
Analyze the following false negative examples for the given prompt:

Current Prompt:
{initial_prompt}

False Negative Examples:
{fn_texts}

Your task is to analyze these false negatives and suggest improvements to the prompt that would increase recall and reduce missed positives. Focus on:
1. Common patterns or characteristics in these false negatives
2. Subtle indicators of positive cases that the current prompt might be missing
3. Concrete suggestions to modify the prompt to capture these missed positives while maintaining overall accuracy

Provide your analysis and suggestions in a clear, structured format.
"""

TRUE_POSITIVES_ANALYSIS_PROMPT = """
Analyze the following true positive examples for the given prompt:

Current Prompt:
{initial_prompt}

True Positive Examples:
{tp_texts}

Your task is to analyze these true positives and identify what makes them successful classifications. Focus on:
1. Common patterns or characteristics in these true positives
2. Key elements of the current prompt that effectively capture these positive cases
3. Suggestions on how to reinforce these successful patterns in the prompt

Provide your analysis and suggestions in a clear, structured format.
"""

# Add this new prompt for analyzing invalid outputs
INVALID_OUTPUTS_ANALYSIS_PROMPT = """
Analyze the following invalid outputs for the given prompt and output format instructions:

Current Prompt:
{initial_prompt}

Output Format Instructions:
{output_format_prompt}

Invalid Outputs:
{invalid_texts}

Your task is to analyze these invalid outputs and suggest improvements to the prompt and output format instructions that would reduce formatting errors. Focus on:
1. Common patterns or characteristics in these invalid outputs
2. Specific elements of the current prompt or output format instructions that might be causing these formatting errors
3. Concrete suggestions to modify the prompt and output format instructions to reduce invalid outputs

Provide your analysis and suggestions in a clear, structured format.
"""

# Update the PROMPT_ENGINEER_INPUT to include the invalid outputs analysis
PROMPT_ENGINEER_INPUT = """
You are an expert in refining prompts for classification tasks. Based on the following analyses and metrics, generate an improved prompt:

## 
Current Prompt:
{initial_prompt}

## 
Output Format Instructions:
{output_format_prompt}

## 
False Positives Analysis:
{fp_analysis}

## 
False Negatives Analysis:
{fn_analysis}

## 
True Positives Analysis:
{tp_analysis}

## 
Invalid Outputs Analysis:
{invalid_analysis}

## 
Previous Metrics:
- Precision: {precision}
- Recall: {recall}
- Accuracy: {accuracy}
- F1 Score: {f1}

## 
Your task is to create an improved prompt that:
1. Addresses the issues identified in the false positives analysis to increase precision
2. Incorporates suggestions from the false negatives analysis to improve recall
3. Reinforces the patterns identified in the true positives analysis
4. Addresses the formatting issues identified in the invalid outputs analysis
5. Maintains or improves overall accuracy and F1 score
6. Is compatible with the given output format instructions

## 
Provide only the new, improved prompt without any additional explanations or headers.
"""