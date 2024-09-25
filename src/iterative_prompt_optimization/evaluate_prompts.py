import re
import json
from .model_interface import get_model_output
from .utils import transform_and_compare_output

def evaluate_prompts(prompts: list, data: list, model: str) -> list:
    """
    Evaluate multiple prompts against a dataset using a specified model.

    Args:
        prompts (list): List of prompts to evaluate
        data (list): List of dictionaries containing 'text' and 'label' keys
        model (str): Name of the model to use for evaluation

    Returns:
        list: List of evaluation results for each prompt
    """
    results = []
    for prompt in prompts:
        prompt_results = []
        for index, row in enumerate(data):
            text = row['text']
            label = row['label']
            
            # Get model output for the current text
            response = get_model_output(prompt, text, index, len(data))
            raw_output = response['choices'][0]['message']['content']
            
            # Define the output schema for transformation and comparison
            output_schema = {
                'key_to_extract': 'risk_output',
                'value_mapping': {
                    'risk present': 1,
                    'risk not present': 0
                },
                'regex_pattern': r"'risk_output':\s*'(.*?)'"
            }
            
            # Transform and compare the model output with the true label
            transformed_output, is_correct, is_valid = transform_and_compare_output(raw_output, label, output_schema)
            
            # Store the results for this text
            prompt_results.append({
                'text': text,
                'label': label,
                'raw_output': raw_output,
                'transformed_output': transformed_output,
                'is_correct': is_correct,
                'is_valid': is_valid
            })
        
        # Add the results for this prompt to the overall results
        results.append(prompt_results)
    
    return results
