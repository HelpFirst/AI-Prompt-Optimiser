# Dashboard generator for binary classification results
# Handles creation of HTML dashboards with metrics, visualizations, and analyses

import os
import json
from jinja2 import Template
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from ast import literal_eval
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

def generate_confusion_matrix(y_true, y_pred):
    """
    Generate a confusion matrix visualization for binary classification.
    
    This function:
    1. Filters out invalid predictions
    2. Creates a confusion matrix visualization
    3. Handles edge cases (no data, errors)
    4. Converts the plot to a base64 encoded image
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        
    Returns:
        str: Base64 encoded image of the confusion matrix
    """
    # Remove pairs where either value is NaN
    valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = np.array(y_true)[valid_indices]
    y_pred_valid = np.array(y_pred)[valid_indices]
    
    if len(y_true_valid) == 0:
        # Handle case with no valid predictions
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        plt.title('Confusion Matrix (No Valid Data)')
        plt.axis('off')
    else:
        # Generate and plot confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_iteration_dashboard(log_dir: str, iteration: int, results: dict, 
                               current_prompt: str, output_format_prompt: str, 
                               initial_prompt: str):
    """
    Generate an HTML dashboard for a single iteration of binary classification.
    
    This function creates a comprehensive dashboard showing:
    - Performance metrics
    - Confusion matrix
    - Detailed prediction results
    - Analysis of different prediction categories
    - Prompt evolution and improvements
    
    Args:
        log_dir: Directory for saving the dashboard
        iteration: Current iteration number
        results: Dictionary containing evaluation results
        current_prompt: Current classification prompt
        output_format_prompt: Output format instructions
        initial_prompt: Original starting prompt
    """
    # Load evaluation results from file
    evaluation_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
    with open(evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Convert evaluation results to DataFrame for easier processing
    df = pd.DataFrame(evaluation_data['evaluations'])
    
    # Try to normalize the raw_output column
    def parse_raw_output(raw_output):
        """Parse raw output into structured format."""
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            try:
                return literal_eval(raw_output)
            except:
                return raw_output

    # Process raw outputs
    df['parsed_output'] = df['raw_output'].apply(parse_raw_output)
    
    try:
        # Normalize parsed JSON outputs into columns
        normalized = pd.json_normalize(df['parsed_output'])
        normalized = normalized[[col for col in normalized.columns if col not in df.columns]]
        df = pd.concat([df, normalized], axis=1)
    except:
        # Continue without normalization if it fails
        pass
    
    # Organize columns for display
    ordered_columns = ['text', 'label', 'transformed_output', 'result'] + [
        col for col in df.columns if col not in ['text', 'label', 'transformed_output', 'result']
    ]
    df = df[ordered_columns]
    
    # Convert DataFrame back to list for template
    evaluation_results = df.to_dict('records')
    table_headers = df.columns.tolist()
    
    # Generate confusion matrix visualization
    y_true = df['label'].tolist()
    y_pred = df['transformed_output'].tolist()
    confusion_matrix_image = generate_confusion_matrix(y_true, y_pred)
    
    # Generate HTML content using template
    html_content = template.render(
        iteration=iteration,
        results=results,
        current_prompt=current_prompt,
        evaluation_results=evaluation_results,
        table_headers=table_headers,
        confusion_matrix_image=confusion_matrix_image
    )
    
    # Save dashboard
    with open(os.path.join(log_dir, f'iteration_{iteration}_dashboard.html'), 'w') as f:
        f.write(html_content)

def generate_combined_dashboard(log_dir: str, all_metrics: list, 
                              best_prompt: str, output_format_prompt: str):
    """
    Generate a combined HTML dashboard for all iterations.
    
    This function creates a summary dashboard showing:
    - Metrics across all iterations
    - Performance trends
    - Best performing prompt
    - Detailed iteration-by-iteration analysis
    
    Args:
        log_dir: Directory for saving the dashboard
        all_metrics: List of metrics from all iterations
        best_prompt: Best performing prompt
        output_format_prompt: Output format instructions
    """
    # Find the best metrics (highest F1 score)
    best_metrics = max(all_metrics, key=lambda x: x['f1'])
    best_iteration = best_metrics['iteration']
    
    # Prepare data for Plotly visualizations
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    colors = ['red', 'blue', 'green', 'purple']
    metrics_data = []
    
    # Create line plots for each metric
    for i, metric in enumerate(metrics):
        metrics_data.append({
            'x': [m['iteration'] for m in all_metrics],
            'y': [m[metric] for m in all_metrics],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': metric.capitalize(),
            'line': {'color': colors[i]}
        })
    
    # Create bar plots for prediction validity
    validity_data = [
        {
            'x': [m['iteration'] for m in all_metrics],
            'y': [m['valid_predictions'] for m in all_metrics],
            'type': 'bar',
            'name': 'Valid Predictions',
            'marker': {'color': 'green'}
        },
        {
            'x': [m['iteration'] for m in all_metrics],
            'y': [m['invalid_predictions'] for m in all_metrics],
            'type': 'bar',
            'name': 'Invalid Predictions',
            'marker': {'color': 'red'}
        }
    ]
    
    # Calculate max values for highlighting in the table
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
    
    # Collect data for all iterations
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
        iteration_file = os.path.join(log_dir, f'iteration_{i+1}_prompt_generation.json')
        if os.path.exists(iteration_file):
            with open(iteration_file, 'r') as f:
                iteration_json = json.load(f)
                iteration_data.update({
                    'prompt': iteration_json['initial_prompt'],
                    'fp_analysis': iteration_json['false_positives_analysis'],
                    'fn_analysis': iteration_json['false_negatives_analysis'],
                    'tp_analysis': iteration_json['true_positives_analysis'],
                    'invalid_analysis': iteration_json['invalid_outputs_analysis'],
                })
        
        iterations.append(iteration_data)
    
    # Generate HTML content using template
    html_content = template.render(
        metrics_data=metrics_data,
        validity_data=validity_data,
        best_prompt=best_prompt,
        output_format_prompt=output_format_prompt,
        iterations=iterations,
        best_metrics=best_metrics,
        best_iteration=best_iteration,
        all_metrics=all_metrics,
        max_values=max_values,
        min_values=min_values
    )
    
    # Save the combined dashboard
    with open(os.path.join(log_dir, 'combined_dashboard.html'), 'w') as f:
        f.write(html_content)
