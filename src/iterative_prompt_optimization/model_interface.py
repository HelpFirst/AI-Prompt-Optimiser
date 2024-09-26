import os
import ollama
import openai
from openai import AzureOpenAI
import anthropic
import google.generativeai as genai
from . import config
import json
import hashlib
from pathlib import Path

# Add this at the top of the file, after imports
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 3

def get_cache_key(full_prompt: str, text: str, model: str) -> str:
    """Generate a unique cache key based on the prompt, text, and model."""
    combined = f"{full_prompt}|{text}|{model}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_output(cache_key: str) -> dict:
    """Retrieve cached output if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with cache_file.open('r') as f:
            return json.load(f)
    return None

def cache_output(cache_key: str, output: dict):
    """Cache the model output."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with cache_file.open('w') as f:
        json.dump(output, f)

def get_model_output(full_prompt: str, text: str, index: int, total: int, use_json_mode: bool = False, use_cache: bool = True) -> dict:
    """
    Get the output from the selected AI model for evaluation.

    Args:
        full_prompt (str): The complete prompt including instructions
        text (str): The input text to be evaluated
        index (int): Current index in the dataset
        total (int): Total number of samples
        use_json_mode (bool): Whether to use JSON mode for output
        use_cache (bool): Whether to use cached results

    Returns:
        dict: Model output containing the generated content
    """
    EVAL_PROVIDER, EVAL_MODEL = config.get_eval_model()
    print(f"Processing text {index + 1}/{total}")
    
    if use_cache:
        cache_key = get_cache_key(full_prompt, text, EVAL_MODEL)
        cached_output = get_cached_output(cache_key)
        if cached_output:
            print(f"Using cached output for text {index + 1}/{total}")
            return cached_output

    for retry in range(MAX_RETRIES):
        try:
            if EVAL_PROVIDER == "ollama":
                output = _get_ollama_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "openai":
                output = _get_openai_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "azure_openai":
                output = _get_azure_openai_output(EVAL_MODEL, full_prompt, text, use_json_mode)
            elif EVAL_PROVIDER == "anthropic":
                output = _get_anthropic_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "google":
                output = _get_google_output(EVAL_MODEL, full_prompt, text)
            
            if use_cache:
                cache_output(cache_key, output)
            return output
        except Exception as e:
            print(f"API error for text at index {index}: {str(e)}. Retrying... (Attempt {retry + 1}/{MAX_RETRIES})")
    print(f"Max retries reached. Using default prediction.")
    return {'choices': [{'message': {'content': '0'}}]}  # Default prediction

def _get_ollama_output(model_name: str, full_prompt: str, text: str) -> dict:
    """Helper function to get output from Ollama models."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': full_prompt},
            {'role': 'user', 'content': text},
        ],
        stream=False
    )
    if 'message' in response and 'content' in response['message']:
        return {'choices': [{'message': {'content': response['message']['content'].strip()}}]}
    else:
        raise ValueError("Unexpected Ollama response format")

def _get_openai_output(model_name: str, full_prompt: str, text: str) -> dict:
    """Helper function to get output from OpenAI models."""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ]
    )
    return {'choices': [{'message': {'content': response.choices[0].message['content'].strip()}}]}

def _get_azure_openai_output(model_name: str, full_prompt: str, text: str, use_json_mode: bool = False) -> dict:
    """Helper function to get output from Azure OpenAI models."""
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,  
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_API_BASE
    )
    deployment_name = config.get_azure_deployment_name(model_name)
    
    completion_args = {
        "model": deployment_name,
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,  # You can adjust this value as needed
    }
    
    if use_json_mode:
        completion_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**completion_args)
    return {'choices': [{'message': {'content': response.choices[0].message.content.strip()}}]}

def _get_anthropic_output(model_name: str, full_prompt: str, text: str) -> dict:
    """Helper function to get output from Anthropic models."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model=model_name,
        prompt=f"{anthropic.HUMAN_PROMPT} {full_prompt}\n\n{text}{anthropic.AI_PROMPT}",
        max_tokens_to_sample=100
    )
    return {'choices': [{'message': {'content': response.completion.strip()}}]}

def _get_google_output(model_name: str, full_prompt: str, text: str) -> dict:
    """Helper function to get output from Google models."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(f"{full_prompt}\n\n{text}")
    return {'choices': [{'message': {'content': response.text.strip()}}]}

def get_analysis(analysis_prompt: str) -> str:
    """
    Get an analysis from the selected AI model for optimization.

    Args:
        analysis_prompt (str): The prompt for generating analysis

    Returns:
        str: Generated analysis
    """
    OPTIM_PROVIDER, OPTIM_MODEL = config.get_optim_model()
    for retry in range(MAX_RETRIES):
        try:
            if OPTIM_PROVIDER == "ollama":
                return _get_ollama_analysis(OPTIM_MODEL, analysis_prompt)
            elif OPTIM_PROVIDER == "openai":
                return _get_openai_analysis(OPTIM_MODEL, analysis_prompt)
            elif OPTIM_PROVIDER == "azure_openai":
                return _get_azure_openai_analysis(OPTIM_MODEL, analysis_prompt)
            elif OPTIM_PROVIDER == "anthropic":
                return _get_anthropic_analysis(OPTIM_MODEL, analysis_prompt)
            elif OPTIM_PROVIDER == "google":
                return _get_google_analysis(OPTIM_MODEL, analysis_prompt)
        except Exception as e:
            print(f"API error during analysis: {e}. Retrying... (Attempt {retry + 1}/{MAX_RETRIES})")
    print(f"Max retries reached. Using default analysis.")
    return "Unable to generate analysis due to API errors."

def _get_ollama_analysis(model_name: str, analysis_prompt: str) -> str:
    """Helper function to get analysis from Ollama models."""
    analysis_response = ollama.chat(model=model_name, messages=[
        {'role': 'user', 'content': analysis_prompt},
    ])
    return analysis_response['message']['content'].strip()

def _get_azure_openai_analysis(model_name: str, analysis_prompt: str) -> str:
    """Helper function to get analysis from Azure OpenAI models."""
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,  
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_API_BASE
    )
    deployment_name = config.get_azure_deployment_name(model_name)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": analysis_prompt},
        ],
        temperature=0.7,  # You can adjust this value as needed
    )
    return response.choices[0].message.content.strip()

