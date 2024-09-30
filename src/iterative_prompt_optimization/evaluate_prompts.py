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

def analyze_errors(model, prompt, fp_examples, fn_examples):
    fp_analysis = ""
    fn_analysis = ""
    
    if fp_examples:
        fp_prompt = f"""Analyze the following false positive examples for the given prompt:

Prompt: {prompt}

False Positive Examples:
{json.dumps(fp_examples, indent=2)}

Provide a detailed analysis of why these examples were incorrectly classified as positive. Include a step-by-step explanation for each example.

Output your analysis as a JSON object with the following structure:
{{
    "fp_text": "Your overall analysis text here",
    "step_by_step": [
        {{
            "example": "The specific example text",
            "explanation": "Step-by-step explanation for this example"
        }},
        ...
    ]
}}
"""
        fp_response = model.generate(fp_prompt)
        fp_analysis = json.loads(fp_response)

    if fn_examples:
        fn_prompt = f"""Analyze the following false negative examples for the given prompt:

Prompt: {prompt}

False Negative Examples:
{json.dumps(fn_examples, indent=2)}

Provide a detailed analysis of why these examples were incorrectly classified as negative. Include a step-by-step explanation for each example.

Output your analysis as a JSON object with the following structure:
{{
    "fn_text": "Your overall analysis text here",
    "step_by_step": [
        {{
            "example": "The specific example text",
            "explanation": "Step-by-step explanation for this example"
        }},
        ...
    ]
}}
"""
        fn_response = model.generate(fn_prompt)
        fn_analysis = json.loads(fn_response)

    combined_analysis = {
        "false_positives": fp_analysis,
        "false_negatives": fn_analysis
    }

    analysis_prompt = f"""Based on the following analyses of false positives and false negatives:

False Positives Analysis:
{json.dumps(fp_analysis, indent=2)}

False Negatives Analysis:
{json.dumps(fn_analysis, indent=2)}

Provide a comprehensive analysis of the prompt's performance, including:
1. Common patterns or issues in false positives and false negatives
2. Potential improvements to the prompt
3. Any other relevant observations

Output your analysis as a JSON object with the following structure:
{{
    "overall_analysis": "Your comprehensive analysis text here",
    "common_patterns": ["List of common patterns or issues"],
    "potential_improvements": ["List of potential improvements"],
    "additional_observations": ["List of any other relevant observations"]
}}
"""

    final_analysis = model.generate(analysis_prompt)
    return json.loads(final_analysis)
