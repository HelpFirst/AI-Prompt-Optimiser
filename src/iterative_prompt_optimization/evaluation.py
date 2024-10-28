import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from .model_interface import get_model_output
from .utils import transform_and_compare_output, create_log_file_path, initialize_log_data, log_results

def evaluate_prompt(full_prompt: str, eval_data: pd.DataFrame, output_schema: dict, 
                   problem_type: str,
                   log_dir: str = None, 
                   iteration: int = None, 
                   use_cache: bool = True, 
                   provider: str = None, 
                   model: str = None, 
                   temperature: float = 0.7) -> dict:
    # Initialize lists
    predictions = []  # Only declare this once
    true_labels = []
    invalid_predictions = 0
    valid_predictions = 0
    false_positives = []
    false_negatives = []
    true_positives = []
    invalid_outputs = []
    raw_outputs = []
    texts = []
    labels = []
    
    log_data = initialize_log_data(full_prompt) if log_dir else None
    use_json_mode = output_schema.get('use_json_mode', False)
    
    for index, row in eval_data.iterrows():
        # Get model output for the current text
        model_output = get_model_output(provider, model, temperature, full_prompt, row['text'], 
                                        index, len(eval_data), use_json_mode, use_cache)
        raw_output = model_output['choices'][0]['message']['content']
        # Transform and compare the model output with the true label
        transformed_output, is_correct, is_valid, chain_of_thought = transform_and_compare_output(raw_output, row['label'], output_schema)
        
        # Process and display output
        result = process_output(transformed_output, row['label'], is_valid, index, len(eval_data), raw_output)
        
        # Store the additional information for every example
        raw_outputs.append(raw_output)
        texts.append(row['text'])
        labels.append(row['label'])
        
        if is_valid:
            valid_predictions += 1
            predictions.append(transformed_output)
            true_labels.append(row['label'])
            if is_correct:
                if transformed_output == 1:
                    true_positives.append({'text': row['text'], 'label': row['label'], 'chain_of_thought': chain_of_thought})
            else:
                if transformed_output == 1 and row['label'] == 0:
                    false_positives.append({'text': row['text'], 'label': row['label'], 'chain_of_thought': chain_of_thought})
                elif transformed_output == 0 and row['label'] == 1:
                    false_negatives.append({'text': row['text'], 'label': row['label'], 'chain_of_thought': chain_of_thought})
        else:
            invalid_predictions += 1
            invalid_outputs.append({'text': row['text'], 'label': row['label'], 'raw_output': raw_output})
        
        if log_data:
            log_data["evaluations"].append({
                "text": row['text'],
                "label": row['label'],
                "raw_output": raw_output,
                "transformed_output": transformed_output,
                "is_correct": is_correct,
                "is_valid": is_valid,
                "chain_of_thought": chain_of_thought,
                "result": result
            })

    # Calculate metrics
    results = calculate_metrics(
        predictions, true_labels, invalid_predictions, valid_predictions,
        false_positives, false_negatives, true_positives, invalid_outputs,
        problem_type
    )
    
    # Add the additional information to results
    results.update({
        'raw_outputs': raw_outputs,
        'texts': texts,
        'labels': labels,
        'predictions': predictions  # Use the same predictions list
    })

    if log_dir and iteration:
        log_file_path = create_log_file_path(log_dir, iteration)
        log_results(log_file_path, log_data, results)

    return results

def create_full_prompt(prompt: str, output_format_prompt: str) -> str:
    """Combine the main prompt and output format prompt."""
    return f"{prompt}\n\n{output_format_prompt}"

def process_output(output: int, ground_truth: int, is_valid: bool, index: int, total: int, raw_output: str) -> str:
    """
    Process the model output and compare it with the ground truth.

    Args:
        output (int): Transformed model output
        ground_truth (int): True label
        is_valid (bool): Whether the output is valid
        index (int): Current index in the dataset
        total (int): Total number of samples
        raw_output (str): Raw output from the model

    Returns:
        str: A string representation of the result
    """
    if not is_valid:
        # Use string concatenation instead of f-string for the part with backslashes
        result = "🛠️ (Invalid Output Format) - Raw output: " + raw_output.replace('\n', '').replace('\r', '')
    elif output == ground_truth:
        result = "✅ (TP)" if output == 1 else "✅ (TN)"
    else:
        result = "❌ (FP)" if output == 1 else "❌ (FN)"
    
    print(f"Prediction {index + 1}/{total}: {output} | Ground Truth: {ground_truth} {result}")
    return result

def calculate_metrics(predictions: list, true_labels: list, invalid_predictions: int, 
                     valid_predictions: int, false_positives: list, false_negatives: list, 
                     true_positives: list, invalid_outputs: list, problem_type: str) -> dict:
    """
    Calculate evaluation metrics for both binary and multiclass classification.
    """
    if len(predictions) > 0:
        # Calculate confusion matrix
        try:
            conf_matrix = confusion_matrix(true_labels, predictions)
            conf_matrix = conf_matrix.tolist() # Convert confusion matrix to list for JSON serialization
        except Exception as e:
            print(f"Warning: Could not calculate confusion matrix: {str(e)}")
            conf_matrix = None
        
        if problem_type == 'binary':
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
        else:
            # For multiclass, use weighted average
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        accuracy = accuracy_score(true_labels, predictions)
    else:
        precision = recall = accuracy = f1 = 0
        conf_matrix = None

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'invalid_predictions': invalid_predictions,
        'valid_predictions': valid_predictions,
        'invalid_outputs': invalid_outputs,
        'problem_type': problem_type,
        'invalid_output_message': (
            "Note: There were invalid output formats detected. Please ensure that the output format "
            "is correct and follows the instructions provided in the prompt. Invalid outputs are not "
            "included in the analysis."
        ) if invalid_predictions > 0 else None
    }
