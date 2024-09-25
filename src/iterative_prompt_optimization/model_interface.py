import os
import ollama
import openai
from openai import AzureOpenAI
import anthropic
import google.generativeai as genai
from . import config

MAX_RETRIES = 3

def get_model_output(full_prompt: str, text: str, index: int, total: int) -> dict:
    """
    Get the output from the selected AI model for evaluation.

    Args:
        full_prompt (str): The complete prompt including instructions
        text (str): The input text to be evaluated
        index (int): Current index in the dataset
        total (int): Total number of samples

    Returns:
        dict: Model output containing the generated content
    """
    EVAL_PROVIDER, EVAL_MODEL = config.get_eval_model()
    print(f"Processing text {index + 1}/{total}")
    
    for retry in range(MAX_RETRIES):
        try:
            if EVAL_PROVIDER == "ollama":
                return _get_ollama_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "openai":
                return _get_openai_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "azure_openai":
                return _get_azure_openai_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "anthropic":
                return _get_anthropic_output(EVAL_MODEL, full_prompt, text)
            elif EVAL_PROVIDER == "google":
                return _get_google_output(EVAL_MODEL, full_prompt, text)
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

def _get_azure_openai_output(model_name: str, full_prompt: str, text: str) -> dict:
    """Helper function to get output from Azure OpenAI models."""
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,  
        api_version=config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_API_BASE
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": text}
        ]
    )
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

# ... (similar helper functions for other providers)