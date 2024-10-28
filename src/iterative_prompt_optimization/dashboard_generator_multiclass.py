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

def generate_confusion_matrix_multiclass(y_true, y_pred):
    """Generate confusion matrix for multiclass classification."""
    try:
        # Filter out any invalid predictions
        valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                        if true is not None and pred is not None]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if len(y_true_valid) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
            plt.title('Confusion Matrix (No Valid Data)')
            plt.axis('off')
        else:
            cm = confusion_matrix(y_true_valid, y_pred_valid)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
    except Exception as e:
        print(f"Error generating confusion matrix: {str(e)}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        plt.title('Confusion Matrix (Error)')
        plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_iteration_dashboard_multiclass(log_dir: str, iteration: int, results: dict, 
                                         current_prompt: str, output_format_prompt: str, 
                                         initial_prompt: str):
    """Generate and save an HTML dashboard for a single iteration of multiclass classification."""
    
    # Load evaluation results
    evaluation_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
    with open(evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Generate confusion matrix
    y_true = results['labels']
    y_pred = results['predictions']
    confusion_matrix_image = generate_confusion_matrix_multiclass(y_true, y_pred)
    
    # Create evaluation results for display
    evaluation_results = []
    for text, label, pred, cot, raw in zip(
        results['texts'],
        results['labels'],
        results['predictions'],
        results['chain_of_thought'],
        results['raw_outputs']
    ):
        evaluation_results.append({
            'Text': text,
            'True Label': label,
            'Predicted': pred,
            'Chain of Thought': cot,
            'Raw Output': raw,
            'Result': '✅' if label == pred else '❌'
        })
    
    table_headers = ['Text', 'True Label', 'Predicted', 'Chain of Thought', 'Raw Output', 'Result']
    
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multiclass Classification - Iteration {{ iteration }} Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.24/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .metrics { display: flex; justify-content: space-around; margin-bottom: 20px; }
            .metric { text-align: center; }
            .prompt, .analysis { white-space: pre-wrap; background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
            .confusion-matrix { width: 50%; margin: 20px auto; text-align: center; }
            .confusion-matrix img { max-width: 100%; height: auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .modal {
                display: none;
                position: fixed;
                z-index: 1;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                overflow: auto;
                background-color: rgba(0,0,0,0.4);
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 20px;
                border: 1px solid #888;
                width: 80%;
                max-height: 70vh;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Iteration {{ iteration }} Dashboard - Multiclass Classification</h1>
            
            <h2>Evaluation Metrics</h2>
            <div class="metrics">
                <div class="metric"><strong>Precision:</strong> {{ "%.4f"|format(results.precision) }}</div>
                <div class="metric"><strong>Recall:</strong> {{ "%.4f"|format(results.recall) }}</div>
                <div class="metric"><strong>Accuracy:</strong> {{ "%.4f"|format(results.accuracy) }}</div>
                <div class="metric"><strong>F1 Score:</strong> {{ "%.4f"|format(results.f1) }}</div>
                <div class="metric"><strong>Valid Predictions:</strong> {{ results.valid_predictions }}</div>
                <div class="metric"><strong>Invalid Predictions:</strong> {{ results.invalid_predictions }}</div>
            </div>

            <h2>Confusion Matrix</h2>
            <div class="confusion-matrix">
                <img src="data:image/png;base64,{{ confusion_matrix_image }}" alt="Confusion Matrix">
            </div>

            <h2>Current Prompt</h2>
            <pre class="prompt">{{ current_prompt }}</pre>
            
            <h2>Evaluation Results</h2>
            <table id="resultsTable" class="display">
                <thead>
                    <tr>
                        {% for header in table_headers %}
                        <th>{{ header }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for eval in evaluation_results %}
                    <tr>
                        {% for header in table_headers %}
                        <td title="{{ eval[header] }}">{{ eval[header] }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <h2>Analyses</h2>
            
            <h3>Correct Predictions Analysis</h3>
            <h4>Prompt Used:</h4>
            <pre class="prompt">{{ results.prompts_used.correct_prompt }}</pre>
            <h4>Analysis:</h4>
            <pre class="analysis">{{ results.correct_analysis }}</pre>
            
            <h3>Incorrect Predictions Analysis</h3>
            <h4>Prompt Used:</h4>
            <pre class="prompt">{{ results.prompts_used.incorrect_prompt }}</pre>
            <h4>Analysis:</h4>
            <pre class="analysis">{{ results.incorrect_analysis }}</pre>
            
            <h3>Prompt Engineer Input</h3>
            <pre class="prompt">{{ results.prompts_used.prompt_engineer_input }}</pre>
            
            <h3>New Generated Prompt</h3>
            <pre class="prompt">{{ results.new_prompt }}</pre>
            
            {% if results.validation_result %}
            <h3>Validation Result</h3>
            <pre class="analysis">{{ results.validation_result }}</pre>
            {% endif %}
        </div>

        <script>
            $(document).ready(function() {
                $('#resultsTable').DataTable({
                    pageLength: 10,
                    lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                    columnDefs: [{
                        targets: '_all',
                        render: function(data, type, row) {
                            if (type === 'display') {
                                return '<div title="' + data + '">' + data + '</div>';
                            }
                            return data;
                        }
                    }]
                });
            });
        </script>
    </body>
    </html>
    ''')
    
    # Generate HTML content
    html_content = template.render(
        iteration=iteration,
        results=results,
        current_prompt=current_prompt,
        evaluation_results=evaluation_results,
        table_headers=table_headers,
        confusion_matrix_image=confusion_matrix_image
    )
    
    # Save dashboard
    with open(os.path.join(log_dir, f'iteration_{iteration}_dashboard_multiclass.html'), 'w') as f:
        f.write(html_content)

def generate_combined_dashboard_multiclass(log_dir: str, all_metrics: list, 
                                        best_prompt: str, output_format_prompt: str):
    """Generate and save a combined HTML dashboard for all iterations of multiclass classification."""
    # Find the best metrics (highest F1 score)
    best_metrics = max(all_metrics, key=lambda x: x['f1'])
    best_iteration = best_metrics['iteration']
    
    # Prepare data for Plotly
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    colors = ['red', 'blue', 'green', 'purple']
    metrics_data = []
    
    for i, metric in enumerate(metrics):
        metrics_data.append({
            'x': [m['iteration'] for m in all_metrics],
            'y': [m[metric] for m in all_metrics],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': metric.capitalize(),
            'line': {'color': colors[i]}
        })
    
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
    
    # Generate and save the combined dashboard HTML
    # (Similar to the binary version but adapted for multiclass metrics)
    # ... implementation continues with the combined dashboard template ...
