# Module for interfacing with various AI model providers
# Handles API calls, caching, and error handling for different model providers

import os
import ollama
import openai
from openai import AzureOpenAI, OpenAI
import anthropic
import google.generativeai as genai
from . import config
import json
import hashlib
from pathlib import Path
import warnings

# Directory for caching model responses to reduce API calls
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Maximum number of retries for API calls before falling back to default behavior
MAX_RETRIES = 3

def get_model_output(provider: str, model: str, temperature: float, full_prompt: str, text: str, 
                    index: int = None, total: int = None, use_json_mode: bool = False, 
                    use_cache: bool = True) -> dict:
    """
    Get output from the selected AI model with caching and error handling.
    
    This function:
    1. Checks cache for existing responses
    2. Makes API calls to the appropriate provider
    3. Handles retries and errors
    4. Caches successful responses
    
    Args:
        provider: AI provider (e.g., "ollama", "openai")
        model: Specific model name
        temperature: Temperature setting for generation
        full_prompt: Complete prompt including instructions
        text: Input text for evaluation
        index: Current index in dataset (for progress tracking)
        total: Total number of samples
        use_json_mode: Whether to enforce JSON output format
        use_cache: Whether to use cached results
        
    Returns:
        dict: Model output containing generated content
    """
    if index is not None and total is not None:
        print("-----------------------------------")
        print(f"Processing text {index + 1}/{total} .....")
    
    # Check cache first if enabled
    if use_cache:
        cache_key = get_cache_key(full_prompt, text, model)
        cached_output = get_cached_output(cache_key)
        if cached_output:
            if index is not None and total is not None:
                print(f"Using cached output for text {index + 1}/{total}")
            return cached_output

    last_error = None
    # Try API calls with retries
    for retry in range(MAX_RETRIES):
        try:
            # Route to appropriate provider-specific function
            if provider == "azure_openai":
                output = _get_azure_openai_output(model, full_prompt, text, use_json_mode, temperature)
            elif provider == "ollama":
                output = _get_ollama_output(model, full_prompt, text, temperature)
            elif provider == "openai":
                output = _get_openai_output(model, full_prompt, text, use_json_mode, temperature)
            elif provider == "anthropic":
                output = _get_anthropic_output(model, full_prompt, text, temperature)
            elif provider == "google":
                output = _get_google_output(model, full_prompt, text, temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache successful response if enabled
            if use_cache:
                cache_output(cache_key, output)
            return output
        except Exception as e:
            last_error = e
            print(f"API error: {str(e)}. Retrying... (Attempt {retry + 1}/{MAX_RETRIES})")
            
    # Return default prediction if all retries fail
    error_msg = f"Failed to get model output after {MAX_RETRIES} retries. Last error: {str(last_error)}"
    raise RuntimeError(error_msg)

# Provider-specific helper functions
def _get_ollama_output(model: str, full_prompt: str, text: str, temperature: float) -> dict:
    """Handle API calls to Ollama models."""
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': full_prompt},
            {'role': 'user', 'content': text},
        ],
        stream=False,
        options={"temperature": temperature}
    )
    if 'message' in response and 'content' in response['message']:
        return {'choices': [{'message': {'content': response['message']['content'].strip()}}]}
    else:
        raise ValueError("Unexpected Ollama response format")

def _get_openai_output(model: str, full_prompt: str, text: str, use_json_mode: bool, temperature: float) -> dict:
    """Helper function to get output from OpenAI models."""
    client = OpenAI()  # Initialize the client
    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": text}
    ]
    completion_args = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    if use_json_mode:
        completion_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**completion_args)
    return {'choices': [{'message': {'content': response.choices[0].message.content.strip()}}]}

def _get_azure_openai_output(model: str, full_prompt: str, text: str, use_json_mode: bool, temperature: float) -> dict:
    """Helper function to get output from Azure OpenAI models."""
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,  
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_API_BASE
    )
    deployment_name = config.get_azure_deployment_name(model)
    
    completion_args = {
        "model": deployment_name,
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature,
    }
    
    if use_json_mode:
        completion_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**completion_args)
    return {'choices': [{'message': {'content': response.choices[0].message.content.strip()}}]}

def _get_anthropic_output(model: str, full_prompt: str, text: str, temperature: float) -> dict:
    """Helper function to get output from Anthropic models."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model=model,
        prompt=f"{anthropic.HUMAN_PROMPT} {full_prompt}\n\n{text}{anthropic.AI_PROMPT}",
        max_tokens_to_sample=100,
        temperature=temperature
    )
    return {'choices': [{'message': {'content': response.completion.strip()}}]}

def _get_google_output(model: str, full_prompt: str, text: str, temperature: float) -> dict:
    """Helper function to get output from Google models."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model)
    
    try:
        response = model.generate_content(
            f"{full_prompt}\n\n{text}",
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return {'choices': [{'message': {'content': response.text.strip()}}]}
    except ValueError as e:
        if "Temperature must be" in str(e):
            warnings.warn(f"Google AI doesn't support the provided temperature value. Using default temperature. Error: {e}")
            response = model.generate_content(f"{full_prompt}\n\n{text}")
            return {'choices': [{'message': {'content': response.text.strip()}}]}
        else:
            raise

def get_analysis(provider: str, model: str, temperature: float, analysis_prompt: str) -> str:
    """
    Get an analysis from the selected AI model for optimization.
    This function does not use JSON mode.
    """
    for retry in range(MAX_RETRIES):
        try:
            if provider == "azure_openai":
                return _get_azure_openai_analysis(model, analysis_prompt, temperature)
            elif provider == "ollama":
                return _get_ollama_analysis(model, analysis_prompt, temperature)
            elif provider == "openai":
                return _get_openai_analysis(model, analysis_prompt, temperature)
            elif provider == "anthropic":
                return _get_anthropic_analysis(model, analysis_prompt, temperature)
            elif provider == "google":
                return _get_google_analysis(model, analysis_prompt, temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            print(f"API error during analysis: {e}. Retrying... (Attempt {retry + 1}/{MAX_RETRIES})")
    print(f"Max retries reached. Using default analysis.")
    return "Unable to generate analysis due to API errors."

def _get_ollama_analysis(model: str, analysis_prompt: str, temperature: float) -> str:
    """Helper function to get analysis from Ollama models."""
    analysis_response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': analysis_prompt}],
        options={"temperature": temperature}
    )
    return analysis_response['message']['content'].strip()

def _get_azure_openai_analysis(model: str, analysis_prompt: str, temperature: float) -> str:
    """Helper function to get analysis from Azure OpenAI models without JSON mode."""
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,  
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_API_BASE
    )
    deployment_name = config.get_azure_deployment_name(model)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": analysis_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def _get_openai_analysis(model: str, analysis_prompt: str, temperature: float) -> str:
    """Helper function to get analysis from OpenAI models."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": analysis_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def _get_anthropic_analysis(model: str, analysis_prompt: str, temperature: float) -> str:
    """Helper function to get analysis from Anthropic models."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model=model,
        prompt=f"{anthropic.HUMAN_PROMPT} {analysis_prompt}{anthropic.AI_PROMPT}",
        max_tokens_to_sample=1000,  # Increased for analysis
        temperature=temperature
    )
    return response.completion.strip()

def _get_google_analysis(model: str, analysis_prompt: str, temperature: float) -> str:
    """Helper function to get analysis from Google models."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model)
    
    try:
        response = model.generate_content(
            analysis_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text.strip()
    except ValueError as e:
        if "Temperature must be" in str(e):
            warnings.warn(f"Google AI doesn't support the provided temperature value. Using default temperature. Error: {e}")
            response = model.generate_content(analysis_prompt)
            return response.text.strip()
        else:
            raise

# Caching helper functions
def get_cache_key(full_prompt: str, text: str, model: str) -> str:
    """Generate a unique cache key based on input parameters."""
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
    """Cache the model output for future use."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with cache_file.open('w') as f:
        json.dump(output, f)
