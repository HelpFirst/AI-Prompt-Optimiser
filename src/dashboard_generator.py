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
    COMBINED_TEMPLATE,
    COMMON_STYLES
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

def get_prediction_type(true_label: int, predicted_label: int) -> str:
    """Get the prediction type (TP, TN, FP, FN) for binary classification."""
    if true_label == predicted_label:
        return "TP" if true_label == 1 else "TN"
    else:
        return "FP" if predicted_label == 1 else "FN"

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
        
        # Try to load prompt generation data, use defaults if not found
        prompt_data = {}
        prompt_gen_path = os.path.join(log_dir, f'iteration_{iteration}_prompt_generation.json')
        
        if os.path.exists(prompt_gen_path):
            with open(prompt_gen_path, 'r') as f:
                prompt_data = json.load(f)
                if prompt_data.get('new_prompt'):
                    last_valid_prompt = prompt_data['new_prompt']
                
                # Extract analysis results based on classification type
                if is_binary:
                    if 'analysis_results' in prompt_data:
                        results['tp_analysis'] = prompt_data['analysis_results'].get('true_positives', '')
                        results['fp_analysis'] = prompt_data['analysis_results'].get('false_positives', '')
                        results['fn_analysis'] = prompt_data['analysis_results'].get('false_negatives', '')
                        results['invalid_analysis'] = prompt_data['analysis_results'].get('invalid_outputs', '')
                else:
                    # For multiclass, use correct/incorrect analysis
                    results['correct_analysis'] = prompt_data.get('correct_analysis', '')
                    results['incorrect_analysis'] = prompt_data.get('incorrect_analysis', '')
                    results['invalid_analysis'] = prompt_data.get('invalid_analysis', '')
                    # Also try alternate locations for the analysis
                    if not results['correct_analysis']:
                        results['correct_analysis'] = prompt_data.get('analysis_results', {}).get('correct_analysis', '')
                    if not results['incorrect_analysis']:
                        results['incorrect_analysis'] = prompt_data.get('analysis_results', {}).get('incorrect_analysis', '')
                    if not results['invalid_analysis']:
                        results['invalid_analysis'] = prompt_data.get('analysis_results', {}).get('invalid_outputs', '')
        
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
                # For binary classification, include prediction type
                pred_type = get_prediction_type(label, pred) if pred is not None else "Invalid"
                result = {
                    'Text': text,
                    'True Label': 'Positive (1)' if label == 1 else 'Negative (0)',
                    'Predicted': 'Positive (1)' if pred == 1 else 'Negative (0)' if pred == 0 else 'Invalid',
                    'Chain of Thought': cot,
                    'Raw Output': raw,
                    'Result': f"{'✅' if label == pred else '❌'} ({pred_type})"
                }
            else:
                result = {
                    'Text': text,
                    'True Label': str(label),
                    'Predicted': str(pred) if pred is not None else 'Invalid',
                    'Result': '✅' if label == pred else '❌',
                    'Chain of Thought': cot,
                    'Raw Output': raw
                }
            evaluation_results.append(result)
        
        table_headers = ['Text', 'True Label', 'Predicted', 'Result', 'Chain of Thought', 'Raw Output']
        
        # Prepare template data
        template_data = {
            'iteration': iteration,
            'results': {
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'accuracy': results.get('accuracy'),
                'f1': results.get('f1'),
                'valid_predictions': results.get('valid_predictions'),
                'invalid_predictions': results.get('invalid_predictions'),
                # Include analysis based on classification type
                **(
                    {
                        'tp_analysis': results.get('tp_analysis', ''),
                        'tn_analysis': results.get('tn_analysis', ''),
                        'fp_analysis': results.get('fp_analysis', ''),
                        'fn_analysis': results.get('fn_analysis', ''),
                        'invalid_analysis': results.get('invalid_analysis', '')
                    } if is_binary else {
                        'correct_analysis': results.get('correct_analysis', ''),
                        'incorrect_analysis': results.get('incorrect_analysis', ''),
                        'invalid_analysis': results.get('invalid_analysis', '')
                    }
                )
            },
            'current_prompt': current_prompt,
            'evaluation_results': evaluation_results,
            'table_headers': table_headers,
            'confusion_matrix_image': confusion_matrix_image,
            'is_binary': is_binary,  # Pass this to the template
            'COMMON_STYLES': COMMON_STYLES  # Add COMMON_STYLES to template data
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
        # Ensure each metrics dict has an iteration number
        for i, metrics in enumerate(all_metrics):
            if 'iteration' not in metrics:
                metrics['iteration'] = i + 1  # Add iteration number if missing
        
        # Find the best metrics (highest F1 score)
        best_metrics = max(all_metrics, key=lambda x: x.get('f1', 0))
        best_iteration = best_metrics.get('iteration', 1)
        
        # Load prompts for each iteration
        for metrics in all_metrics:
            iteration = metrics.get('iteration')
            if iteration:
                eval_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                        metrics['prompt'] = eval_data.get('prompt', 'N/A')
        
        # Calculate max/min values for highlighting
        max_values = {
            'precision': max(m.get('precision', 0) for m in all_metrics),
            'recall': max(m.get('recall', 0) for m in all_metrics),
            'accuracy': max(m.get('accuracy', 0) for m in all_metrics),
            'f1': max(m.get('f1', 0) for m in all_metrics),
            'valid_predictions': max(m.get('valid_predictions', 0) for m in all_metrics),
        }
        min_values = {
            'invalid_predictions': min(m.get('invalid_predictions', float('inf')) for m in all_metrics)
        }

        # Sort metrics by iteration number
        sorted_metrics = sorted(all_metrics, key=lambda x: x.get('iteration', float('inf')))

        # Prepare visualization data
        metrics_data = prepare_metrics_visualization(sorted_metrics)
        validity_data = prepare_validity_visualization(sorted_metrics)
        
        # Collect iteration data
        iterations = []
        for metrics in sorted_metrics:
            iteration_data = {
                'number': metrics.get('iteration'),  # Use the iteration number here
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'accuracy': metrics.get('accuracy'),
                'f1': metrics.get('f1'),
                'prompt': metrics.get('prompt', 'N/A'),
                'valid_predictions': metrics.get('valid_predictions'),
                'invalid_predictions': metrics.get('invalid_predictions')
            }
            
            # Load analysis data
            prompt_gen_file = os.path.join(log_dir, f'iteration_{metrics.get("iteration")}_prompt_generation.json')
            if os.path.exists(prompt_gen_file):
                with open(prompt_gen_file, 'r') as f:
                    prompt_data = json.load(f)
                    if 'analysis_results' in prompt_data:
                        iteration_data.update({
                            'correct_analysis': prompt_data['analysis_results'].get('correct_analysis', ''),
                            'incorrect_analysis': prompt_data['analysis_results'].get('incorrect_analysis', ''),
                            'fp_analysis': prompt_data['analysis_results'].get('false_positives', ''),
                            'fn_analysis': prompt_data['analysis_results'].get('false_negatives', ''),
                            'tp_analysis': prompt_data['analysis_results'].get('true_positives', ''),
                            'invalid_analysis': prompt_data['analysis_results'].get('invalid_outputs', '')
                        })
            
            iterations.append(iteration_data)
        
        # Generate HTML content
        html_content = COMBINED_TEMPLATE.render(
            metrics_data=metrics_data,
            validity_data=validity_data,
            best_prompt=best_prompt,
            iterations=iterations,
            best_metrics=best_metrics,
            best_iteration=best_iteration,
            all_metrics=sorted_metrics,  # Use sorted metrics here
            max_values=max_values,
            min_values=min_values,
            COMMON_STYLES=COMMON_STYLES  # Add COMMON_STYLES to template data
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
        'x': [m.get('iteration', i+1) for i, m in enumerate(all_metrics)],
        'y': [m.get(metric, 0) for m in all_metrics],
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
            'x': [m.get('iteration', i+1) for i, m in enumerate(all_metrics)],
            'y': [m.get('valid_predictions', 0) for m in all_metrics],
            'type': 'bar',
            'name': 'Valid Predictions',
            'marker': {'color': '#2ECC71'}
        },
        {
            'x': [m.get('iteration', i+1) for i, m in enumerate(all_metrics)],
            'y': [m.get('invalid_predictions', 0) for m in all_metrics],
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
    """Collect detailed data for each iteration."""
    iterations = []
    
    for metrics in all_metrics:
        iteration_num = metrics.get('iteration')
        if iteration_num is None:
            continue
            
        iteration_data = {
            'iteration': iteration_num,  # Explicitly include iteration number
            **metrics  # Include all other metrics
        }
            
        # Load evaluation data
        eval_file = os.path.join(log_dir, f'iteration_{iteration_num}_evaluation.json')
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                iteration_data['prompt'] = eval_data.get('prompt', 'N/A')
        
        # Load prompt generation data
        prompt_gen_file = os.path.join(log_dir, f'iteration_{iteration_num}_prompt_generation.json')
        if os.path.exists(prompt_gen_file):
            with open(prompt_gen_file, 'r') as f:
                prompt_data = json.load(f)
                if 'analysis_results' in prompt_data:
                    iteration_data.update({
                        'correct_analysis': prompt_data['analysis_results'].get('correct_analysis', ''),
                        'incorrect_analysis': prompt_data['analysis_results'].get('incorrect_analysis', ''),
                        'fp_analysis': prompt_data['analysis_results'].get('false_positives', ''),
                        'fn_analysis': prompt_data['analysis_results'].get('false_negatives', ''),
                        'tp_analysis': prompt_data['analysis_results'].get('true_positives', ''),
                        'invalid_analysis': prompt_data['analysis_results'].get('invalid_outputs', '')
                    })
                if 'new_prompt' in prompt_data:
                    iteration_data['next_prompt'] = prompt_data['new_prompt']
        
        iterations.append(iteration_data)
    
    return sorted(iterations, key=lambda x: x['iteration'])  # Sort by iteration number

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
