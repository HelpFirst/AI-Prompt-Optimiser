# Module for evaluating prompt performance on classification tasks
# Handles both binary and multiclass classification evaluation

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
    """
    Evaluate a prompt's performance on classification tasks.
    
    This function:
    1. Processes each example in the evaluation dataset
    2. Collects predictions and analyzes errors
    3. Calculates performance metrics
    4. Logs results if logging is enabled
    
    Args:
        full_prompt: Complete prompt to evaluate
        eval_data: Dataset containing texts and labels
        output_schema: Schema for parsing model outputs
        problem_type: Type of classification ("binary" or "multiclass")
        log_dir: Directory for storing logs
        iteration: Current iteration number
        use_cache: Whether to use cached model outputs
        provider: AI provider name
        model: Model name
        temperature: Temperature setting for generation
        
    Returns:
        dict: Comprehensive evaluation results and metrics
    """
    # Initialize tracking lists for predictions and analysis
    predictions = []  
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
    chain_of_thought_list = [] 
    
    # Initialize logging if enabled
    log_data = initialize_log_data(full_prompt) if log_dir else None
    use_json_mode = output_schema.get('use_json_mode', False)
    
    # Process each example in the evaluation dataset
    for index, row in eval_data.iterrows():
        # Get model output for the current text
        model_output = get_model_output(provider, model, temperature, full_prompt, row['text'], 
                                      index, len(eval_data), use_json_mode, use_cache)
        raw_output = model_output['choices'][0]['message']['content']
        
        # Transform and validate the model output
        transformed_output, is_correct, is_valid, chain_of_thought = transform_and_compare_output(
            raw_output, row['label'], output_schema
        )
        
        # Process and display output
        result = process_output(transformed_output, row['label'], is_valid, index, 
                              len(eval_data), raw_output, problem_type)
        
        # Store information for every example
        raw_outputs.append(raw_output)
        texts.append(row['text'])
        labels.append(row['label'])
        chain_of_thought_list.append(chain_of_thought)
        
        if is_valid:
            # Track valid predictions and analyze correctness
            valid_predictions += 1
            predictions.append(transformed_output)
            true_labels.append(row['label'])
            if is_correct:
                if transformed_output == 1:  # For binary classification
                    true_positives.append({
                        'text': row['text'], 
                        'label': row['label'], 
                        'chain_of_thought': chain_of_thought
                    })
            else:
                if transformed_output == 1 and row['label'] == 0:
                    false_positives.append({
                        'text': row['text'], 
                        'label': row['label'], 
                        'chain_of_thought': chain_of_thought
                    })
                elif transformed_output == 0 and row['label'] == 1:
                    false_negatives.append({
                        'text': row['text'], 
                        'label': row['label'], 
                        'chain_of_thought': chain_of_thought
                    })
        else:
            # Track invalid outputs for analysis
            invalid_predictions += 1
            invalid_outputs.append({
                'text': row['text'], 
                'label': row['label'], 
                'raw_output': raw_output
            })
        
        # Log evaluation data if enabled
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

    # Calculate performance metrics
    results = calculate_metrics(
        predictions, true_labels, invalid_predictions, valid_predictions,
        false_positives, false_negatives, true_positives, invalid_outputs,
        problem_type
    )
    
    # Add additional information to results
    results.update({
        'raw_outputs': raw_outputs,
        'texts': texts,
        'labels': labels,
        'predictions': predictions,
        'chain_of_thought': chain_of_thought_list
    })

    # Log results if enabled
    if log_dir and iteration:
        log_file_path = create_log_file_path(log_dir, iteration)
        log_results(log_file_path, log_data, results)

    return results

def process_output(output: int, ground_truth: int, is_valid: bool, index: int, 
                  total: int, raw_output: str, problem_type: str = "binary") -> str:
    """
    Process and format model output for display.
    
    Args:
        output: Transformed model output
        ground_truth: True label
        is_valid: Whether output format is valid
        index: Current example index
        total: Total number of examples
        raw_output: Raw model output
        problem_type: Type of classification task
        
    Returns:
        str: Formatted result string with emoji indicators
    """
    if not is_valid:
        result = "🛠️ (Invalid Output Format) - Raw output: " + raw_output.replace('\n', '').replace('\r', '')
    elif problem_type == "binary":
        # Binary classification specific output
        if output == ground_truth:
            result = "✅ (TP)" if output == 1 else "✅ (TN)"
        else:
            result = "❌ (FP)" if output == 1 else "❌ (FN)"
    else:
        # Multiclass classification output
        result = "✅ (Correct)" if output == ground_truth else "❌ (Incorrect)"
    
    print(f"Prediction {index + 1}/{total}: {output} | Ground Truth: {ground_truth} {result}")
    return result

def calculate_metrics(predictions: list, true_labels: list, invalid_predictions: int, 
                     valid_predictions: int, false_positives: list, false_negatives: list, 
                     true_positives: list, invalid_outputs: list, problem_type: str) -> dict:
    """
    Calculate evaluation metrics for classification results.
    
    Handles both binary and multiclass classification:
    - For binary: calculates precision, recall, F1 with standard thresholds
    - For multiclass: uses weighted averaging for metrics
    
    Args:
        predictions: List of model predictions
        true_labels: List of true labels
        invalid_predictions: Count of invalid outputs
        valid_predictions: Count of valid outputs
        false_positives: List of false positive examples
        false_negatives: List of false negative examples
        true_positives: List of true positive examples
        invalid_outputs: List of invalid output examples
        problem_type: Type of classification task
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    if len(predictions) > 0:
        # Calculate confusion matrix
        try:
            conf_matrix = confusion_matrix(true_labels, predictions)
            conf_matrix = conf_matrix.tolist()  # Convert for JSON serialization
        except Exception as e:
            print(f"Warning: Could not calculate confusion matrix: {str(e)}")
            conf_matrix = None
        
        if problem_type == 'binary':
            # Binary classification metrics
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
        else:
            # Multiclass metrics with weighted averaging
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        accuracy = accuracy_score(true_labels, predictions)
    else:
        # Default values if no valid predictions
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
