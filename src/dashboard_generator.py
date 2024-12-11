"""
Dashboard generator for classification results.
Handles creation of HTML dashboards with metrics, visualizations, and analyses
for both binary and multiclass classification problems.
"""

import os
import json
import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from typing import List, Dict, Union, Optional
from .dashboard_templates import (
    ITERATION_TEMPLATE,
    COMBINED_TEMPLATE
)

def create_error_visualization(error_message: str) -> str:
    """Create a visualization for error cases."""
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f'Error: {error_message}', 
             ha='center', va='center', wrap=True)
    plt.title('Confusion Matrix (Error)')
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_confusion_matrix(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    is_binary: bool = True
) -> str:
    """
    Generate a confusion matrix visualization for classification.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        is_binary: Whether this is a binary classification problem
        
    Returns:
        str: Base64 encoded image of the confusion matrix
    """
    try:
        # Validate inputs
        if not y_true or not y_pred:
            raise ValueError("Empty prediction arrays")
        if len(y_true) != len(y_pred):
            raise ValueError("Mismatched array lengths")

        # Filter out invalid predictions
        valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                        if true is not None and pred is not None]
        
        if not valid_indices:
            return create_error_visualization("No valid predictions")

        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        plt.figure(figsize=(10, 8))
        
        if is_binary:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative (0)', 'Positive (1)'],
                       yticklabels=['Negative (0)', 'Positive (1)'])
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
    except Exception as e:
        return create_error_visualization(str(e))
    
    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_iteration_dashboard(
    log_dir: str,
    iteration: int,
    results: Dict[str, Union[float, List, str]],
    current_prompt: str,
    output_format_prompt: str,
    initial_prompt: str,
    is_binary: bool = True
) -> None:
    """
    Generate an HTML dashboard for a single iteration.
    
    Args:
        log_dir: Directory for saving the dashboard
        iteration: Current iteration number
        results: Dictionary containing evaluation results
        current_prompt: Current classification prompt
        output_format_prompt: Output format instructions
        initial_prompt: Original starting prompt
        is_binary: Whether this is a binary classification problem
    """
    try:
        # Load evaluation results
        evaluation_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
        with open(evaluation_file, 'r') as f:
            evaluation_data = json.load(f)
        
        # Load and process prompt generation results
        prompt_gen_file = os.path.join(log_dir, f'iteration_{iteration}_prompt_generation.json')
        if os.path.exists(prompt_gen_file):
            with open(prompt_gen_file, 'r') as f:
                prompt_gen_data = json.load(f)
                if is_binary:
                    results.update({
                        'fp_analysis': prompt_gen_data.get('analysis_results', {}).get('false_positives', ''),
                        'fn_analysis': prompt_gen_data.get('analysis_results', {}).get('false_negatives', ''),
                        'tp_analysis': prompt_gen_data.get('analysis_results', {}).get('true_positives', ''),
                        'invalid_analysis': prompt_gen_data.get('analysis_results', {}).get('invalid_outputs', ''),
                        'prompts_used': prompt_gen_data.get('prompts_for_analysis', {}),
                        'new_prompt': prompt_gen_data.get('new_prompt', '')
                    })
                else:
                    results.update({
                        'correct_analysis': prompt_gen_data.get('correct_analysis', ''),
                        'incorrect_analysis': prompt_gen_data.get('incorrect_analysis', ''),
                        'prompts_used': results.get('prompts_used', {}),
                        'new_prompt': prompt_gen_data.get('new_prompt', '')
                    })
        
        # Generate confusion matrix
        confusion_matrix_image = generate_confusion_matrix(
            results['labels'],
            results['predictions'],
            is_binary
        )
        
        # Create evaluation results for display
        evaluation_results = []
        for text, label, pred, cot, raw in zip(
            results['texts'],
            results['labels'],
            results['predictions'],
            results['chain_of_thought'],
            results['raw_outputs']
        ):
            if is_binary:
                result = {
                    'Text': text,
                    'True Label': 'Positive (1)' if label == 1 else 'Negative (0)',
                    'Predicted': 'Positive (1)' if pred == 1 else 'Negative (0)',
                    'Chain of Thought': cot,
                    'Raw Output': raw,
                    'Result': '✅' if label == pred else '❌'
                }
            else:
                result = {
                    'Text': text,
                    'True Label': str(label),
                    'Predicted': str(pred),
                    'Chain of Thought': cot,
                    'Raw Output': raw,
                    'Result': '✅' if label == pred else '❌'
                }
            evaluation_results.append(result)
        
        table_headers = ['Text', 'True Label', 'Predicted', 'Chain of Thought', 'Raw Output', 'Result']
        
        # Prepare template data
        template_data = {
            'iteration': iteration,
            'results': results,
            'current_prompt': current_prompt,
            'evaluation_results': evaluation_results,
            'table_headers': table_headers,
            'confusion_matrix_image': confusion_matrix_image
        }
        
        # Generate HTML content
        html_content = ITERATION_TEMPLATE.render(**template_data)
        
        # Save dashboard
        suffix = 'binary' if is_binary else 'multiclass'
        output_file = os.path.join(log_dir, f'iteration_{iteration}_dashboard_{suffix}.html')
        with open(output_file, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        print(f"Error generating iteration dashboard: {str(e)}")
        raise

def generate_combined_dashboard(
    log_dir: str,
    all_metrics: List[Dict[str, Union[float, int]]],
    best_prompt: str,
    output_format_prompt: str,
    is_binary: bool = True
) -> None:
    """
    Generate a combined HTML dashboard for all iterations.
    
    Args:
        log_dir: Directory for saving the dashboard
        all_metrics: List of metrics from all iterations
        best_prompt: Best performing prompt
        output_format_prompt: Output format instructions
        is_binary: Whether this is a binary classification problem
    """
    try:
        # Find the best metrics (highest F1 score)
        best_metrics = max(all_metrics, key=lambda x: x['f1'])
        best_iteration = best_metrics['iteration']
        
        # Prepare visualization data
        metrics_data = prepare_metrics_visualization(all_metrics)
        validity_data = prepare_validity_visualization(all_metrics)
        
        # Calculate max/min values for highlighting
        max_values = {
            'precision': max(m['precision'] for m in all_metrics),
            'recall': max(m['recall'] for m in all_metrics),
            'accuracy': max(m['accuracy'] for m in all_metrics),
            'f1': max(m['f1'] for m in all_metrics),
            'valid_predictions': max(m['valid_predictions'] for m in all_metrics),
        }
        min_values = {
            'invalid_predictions': min(m['invalid_predictions'] for m in all_metrics)
        }

        # Collect iteration data
        iterations = collect_iteration_data(log_dir, all_metrics, is_binary)
        
        # Generate HTML content
        html_content = COMBINED_TEMPLATE.render(
            metrics_data=metrics_data,
            validity_data=validity_data,
            best_prompt=best_prompt,
            iterations=iterations,
            best_metrics=best_metrics,
            best_iteration=best_iteration,
            all_metrics=all_metrics,
            max_values=max_values,
            min_values=min_values
        )
        
        # Save the combined dashboard
        suffix = 'binary' if is_binary else 'multiclass'
        output_file = os.path.join(log_dir, f'combined_dashboard_{suffix}.html')
        with open(output_file, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        print(f"Error generating combined dashboard: {str(e)}")
        raise

def prepare_metrics_visualization(
    all_metrics: List[Dict[str, Union[float, int]]]
) -> List[Dict]:
    """Prepare metrics data for visualization."""
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    return [{
        'x': [m['iteration'] for m in all_metrics],
        'y': [m[metric] for m in all_metrics],
        'type': 'scatter',
        'mode': 'lines+markers',
        'name': metric.capitalize(),
        'line': {'color': colors[i]}
    } for i, metric in enumerate(metrics)]

def prepare_validity_visualization(
    all_metrics: List[Dict[str, Union[float, int]]]
) -> List[Dict]:
    """Prepare validity data for visualization."""
    return [
        {
            'x': [m['iteration'] for m in all_metrics],
            'y': [m['valid_predictions'] for m in all_metrics],
            'type': 'bar',
            'name': 'Valid Predictions',
            'marker': {'color': '#2ECC71'}
        },
        {
            'x': [m['iteration'] for m in all_metrics],
            'y': [m['invalid_predictions'] for m in all_metrics],
            'type': 'bar',
            'name': 'Invalid Predictions',
            'marker': {'color': '#E74C3C'}
        }
    ]

def collect_iteration_data(
    log_dir: str,
    all_metrics: List[Dict[str, Union[float, int]]],
    is_binary: bool
) -> List[Dict]:
    """Collect and process data for each iteration."""
    iterations = []
    for i, metrics in enumerate(all_metrics):
        iteration_data = {
            'number': i + 1,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'valid_predictions': metrics['valid_predictions'],
            'invalid_predictions': metrics['invalid_predictions'],
        }
        
        # Load additional data from iteration files
        prompt_gen_file = os.path.join(log_dir, f'iteration_{i+1}_prompt_generation.json')
        if os.path.exists(prompt_gen_file):
            with open(prompt_gen_file, 'r') as f:
                prompt_data = json.load(f)
                if is_binary:
                    iteration_data.update({
                        'prompt': prompt_data.get('initial_prompt', ''),
                        'fp_analysis': prompt_data.get('analysis_results', {}).get('false_positives', ''),
                        'fn_analysis': prompt_data.get('analysis_results', {}).get('false_negatives', ''),
                        'tp_analysis': prompt_data.get('analysis_results', {}).get('true_positives', ''),
                        'invalid_analysis': prompt_data.get('analysis_results', {}).get('invalid_outputs', ''),
                        'prompts_used': prompt_data.get('prompts_for_analysis', {})
                    })
                else:
                    iteration_data.update({
                        'prompt': prompt_data.get('initial_prompt', ''),
                        'correct_analysis': prompt_data.get('correct_analysis', ''),
                        'incorrect_analysis': prompt_data.get('incorrect_analysis', ''),
                        'prompts_used': prompt_data.get('prompts_for_analysis', {})
                    })
        
        iterations.append(iteration_data)
    
    return iterations

# Backward compatibility functions
def generate_iteration_dashboard_binary(*args, **kwargs):
    """Legacy function for binary classification dashboard generation."""
    return generate_iteration_dashboard(*args, **kwargs, is_binary=True)

def generate_combined_dashboard_binary(*args, **kwargs):
    """Legacy function for binary combined dashboard generation."""
    return generate_combined_dashboard(*args, **kwargs, is_binary=True)

def generate_iteration_dashboard_multiclass(*args, **kwargs):
    """Legacy function for multiclass classification dashboard generation."""
    return generate_iteration_dashboard(*args, **kwargs, is_binary=False)

def generate_combined_dashboard_multiclass(*args, **kwargs):
    """Legacy function for multiclass combined dashboard generation."""
    return generate_combined_dashboard(*args, **kwargs, is_binary=False)
