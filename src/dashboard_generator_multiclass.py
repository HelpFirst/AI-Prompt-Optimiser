# Dashboard generator for multiclass classification results
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

def generate_confusion_matrix_multiclass(y_true, y_pred):
    """
    Generate a confusion matrix visualization for multiclass classification.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        
    Returns:
        str: Base64 encoded image of the confusion matrix
    """
    try:
        # Filter out invalid predictions to ensure matrix generation
        valid_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                        if true is not None and pred is not None]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if len(y_true_valid) == 0:
            # Handle case with no valid predictions
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
            plt.title('Confusion Matrix (No Valid Data)')
            plt.axis('off')
        else:
            # Generate and plot confusion matrix
            cm = confusion_matrix(y_true_valid, y_pred_valid)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
    except Exception as e:
        # Handle any errors during matrix generation
        print(f"Error generating confusion matrix: {str(e)}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        plt.title('Confusion Matrix (Error)')
        plt.axis('off')
    
    # Convert plot to base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_iteration_dashboard_multiclass(log_dir: str, iteration: int, results: dict, 
                                         current_prompt: str, output_format_prompt: str, 
                                         initial_prompt: str):
    """
    Generate an HTML dashboard for a single iteration of multiclass classification.
    
    The dashboard includes:
    - Evaluation metrics
    - Confusion matrix visualization
    - Detailed results table
    - Prompt analysis and improvements
    
    Args:
        log_dir: Directory for saving the dashboard
        iteration: Current iteration number
        results: Dictionary containing evaluation results
        current_prompt: Current classification prompt
        output_format_prompt: Output format instructions
        initial_prompt: Original starting prompt
    """
    
    # Load evaluation results
    evaluation_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
    with open(evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Load prompt generation results if available
    prompt_gen_file = os.path.join(log_dir, f'iteration_{iteration}_prompt_generation.json')
    if os.path.exists(prompt_gen_file):
        with open(prompt_gen_file, 'r') as f:
            prompt_gen_data = json.load(f)
            # Update results with analyses from prompt generation
            results['correct_analysis'] = prompt_gen_data.get('correct_analysis', '')
            results['incorrect_analysis'] = prompt_gen_data.get('incorrect_analysis', '')
            results['prompts_used'] = results.get('prompts_used', {})
            results['new_prompt'] = prompt_gen_data.get('new_prompt', '')
    
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
    
    # Define the HTML template
    template = Template(r'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multiclass Classification - Iteration {{ iteration }} Dashboard</title>
        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
        <script type="text/javascript" src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
            }
            .metrics { 
                display: flex; 
                justify-content: space-around; 
                margin-bottom: 20px; 
            }
            .metric { 
                text-align: center; 
            }
            .prompt, .analysis { 
                white-space: pre-wrap; 
                background-color: #f0f0f0; 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 20px; 
            }
            .confusion-matrix { 
                width: 50%; 
                margin: 20px auto; 
                text-align: center; 
            }
            .confusion-matrix img { 
                max-width: 100%; 
                height: auto; 
            }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 20px; 
            }
            th, td { 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }
            th { 
                background-color: #f2f2f2; 
            }
            .truncate {
                max-width: 200px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
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
            #modalText {
                white-space: pre-wrap;
                word-wrap: break-word;
                max-width: 100%;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
            }
            td { 
                max-width: 200px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                cursor: pointer;
            }
            td:hover {
                background-color: #f5f5f5;
            }
            .dataTables_wrapper {
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .dataTables_filter input {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-left: 10px;
            }
            .dataTables_length select {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
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
                        <td title="Double click to expand">{{ eval[header] }}</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <!-- Add modal dialog -->
            <div id="myModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <pre id="modalText"></pre>
                </div>
            </div>

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
                // Initialize DataTable with advanced features
                var table = $('#resultsTable').DataTable({
                    pageLength: 10,
                    lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 'csv', 'excel'
                    ],
                    order: [[0, 'asc']],
                    columnDefs: [{
                        targets: '_all',
                        render: function(data, type, row) {
                            if (type === 'display') {
                                if (data && data.length > 100) {
                                    return '<span title="Click to view full content">' + 
                                           data.substr(0, 100) + '...</span>';
                                }
                            }
                            return data;
                        }
                    }]
                });

                // Modal functionality
                var modal = document.getElementById("myModal");
                var modalText = document.getElementById("modalText");
                var span = document.getElementsByClassName("close")[0];

                // Click handler for table cells
                $('#resultsTable tbody').on('click', 'td', function() {
                    var cellContent = table.cell(this).data();
                    if (cellContent) {
                        modalText.textContent = cellContent;
                        modal.style.display = "block";
                    }
                });

                // Close modal handlers
                span.onclick = function() {
                    modal.style.display = "none";
                }

                window.onclick = function(event) {
                    if (event.target == modal) {
                        modal.style.display = "none";
                    }
                }

                // Close modal on escape key
                $(document).keydown(function(e) {
                    if (e.key === "Escape") {
                        modal.style.display = "none";
                    }
                });
            });
        </script>
    </body>
    </html>
    ''')
    
    # Prepare template data
    template_data = {
        'iteration': iteration,
        'results': results,  # Pass the entire results dictionary directly
        'current_prompt': current_prompt,
        'evaluation_results': evaluation_results,
        'table_headers': table_headers,
        'confusion_matrix_image': confusion_matrix_image
    }
    
    # Generate HTML content
    html_content = template.render(**template_data)
    
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
    
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Combined Multiclass Experiment Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .prompt { white-space: pre-wrap; background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
            .iteration { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .metrics { display: flex; flex-wrap: wrap; justify-content: space-between; }
            .metric { width: 30%; margin-bottom: 10px; }
            .analysis { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; }
            .best-value { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Combined Multiclass Experiment Dashboard</h1>
            
            <h2>Summary of All Iterations</h2>
            <table>
                <tr>
                    <th>Iteration</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Accuracy</th>
                    <th>F1-score</th>
                    <th>Valid Predictions</th>
                    <th>Invalid Predictions</th>
                </tr>
                {% for metrics in all_metrics %}
                <tr>
                    <td>{{ metrics.iteration }}</td>
                    <td{% if metrics.precision == max_values.precision %} class="best-value"{% endif %}>
                        {{ "%.4f"|format(metrics.precision) }}
                    </td>
                    <td{% if metrics.recall == max_values.recall %} class="best-value"{% endif %}>
                        {{ "%.4f"|format(metrics.recall) }}
                    </td>
                    <td{% if metrics.accuracy == max_values.accuracy %} class="best-value"{% endif %}>
                        {{ "%.4f"|format(metrics.accuracy) }}
                    </td>
                    <td{% if metrics.f1 == max_values.f1 %} class="best-value"{% endif %}>
                        {{ "%.4f"|format(metrics.f1) }}
                    </td>
                    <td{% if metrics.valid_predictions == max_values.valid_predictions %} class="best-value"{% endif %}>
                        {{ metrics.valid_predictions }}
                    </td>
                    <td{% if metrics.invalid_predictions == min_values.invalid_predictions %} class="best-value"{% endif %}>
                        {{ metrics.invalid_predictions }}
                    </td>
                </tr>
                {% endfor %}
            </table>

            <div id="metricsChart"></div>
            <div id="validityChart"></div>

            <h2>Best Performing Prompt (Iteration {{ best_iteration }})</h2>
            <div class="metrics">
                <div class="metric">F1 Score: {{ "%.4f"|format(best_metrics.f1) }}</div>
                <div class="metric">Precision: {{ "%.4f"|format(best_metrics.precision) }}</div>
                <div class="metric">Recall: {{ "%.4f"|format(best_metrics.recall) }}</div>
                <div class="metric">Accuracy: {{ "%.4f"|format(best_metrics.accuracy) }}</div>
            </div>
            <pre class="prompt">{{ best_prompt }}</pre>

            <h2>Output Format Prompt</h2>
            <pre class="prompt">{{ output_format_prompt }}</pre>

            <h2>Detailed Iteration History</h2>
            {% for iteration in iterations %}
            <div class="iteration">
                <h3>Iteration {{ iteration.number }}</h3>
                <div class="metrics">
                    <div class="metric"><strong>Precision:</strong> {{ "%.4f"|format(iteration.precision) }}</div>
                    <div class="metric"><strong>Recall:</strong> {{ "%.4f"|format(iteration.recall) }}</div>
                    <div class="metric"><strong>Accuracy:</strong> {{ "%.4f"|format(iteration.accuracy) }}</div>
                    <div class="metric"><strong>F1 Score:</strong> {{ "%.4f"|format(iteration.f1) }}</div>
                    <div class="metric"><strong>Valid Predictions:</strong> {{ iteration.valid_predictions }}</div>
                    <div class="metric"><strong>Invalid Predictions:</strong> {{ iteration.invalid_predictions }}</div>
                </div>
                
                <h4>Prompt Used</h4>
                <pre class="prompt">{{ iteration.prompt }}</pre>
                
                <div class="analysis">
                    <h4>Correct Predictions Analysis</h4>
                    <pre class="prompt">{{ iteration.correct_analysis }}</pre>
                    
                    <h4>Incorrect Predictions Analysis</h4>
                    <pre class="prompt">{{ iteration.incorrect_analysis }}</pre>
                </div>
            </div>
            {% endfor %}
        </div>

        <script>
            var metricsData = {{ metrics_data|tojson }};
            var metricsLayout = {
                title: 'Metrics Across Iterations',
                xaxis: {title: 'Iteration'},
                yaxis: {title: 'Score', range: [0, 1]}
            };
            Plotly.newPlot('metricsChart', metricsData, metricsLayout);

            var validityData = {{ validity_data|tojson }};
            var validityLayout = {
                title: 'Valid vs Invalid Predictions Across Iterations',
                xaxis: {title: 'Iteration'},
                yaxis: {title: 'Number of Predictions'}
            };
            Plotly.newPlot('validityChart', validityData, validityLayout);
        </script>
    </body>
    </html>
    ''')

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
            'prompt': '',  # Initialize with empty string
            'correct_analysis': '',  # Initialize with empty string
            'incorrect_analysis': ''  # Initialize with empty string
        }
        
        # Load additional data from iteration files
        prompt_gen_file = os.path.join(log_dir, f'iteration_{i+1}_prompt_generation.json')
        eval_file = os.path.join(log_dir, f'iteration_{i+1}_evaluation.json')
        
        if os.path.exists(prompt_gen_file):
            with open(prompt_gen_file, 'r') as f:
                prompt_gen_data = json.load(f)
                # Get prompt from initial_prompt field
                iteration_data['prompt'] = prompt_gen_data.get('initial_prompt', '')
                # Get analyses from correct/incorrect analysis fields
                iteration_data['correct_analysis'] = prompt_gen_data.get('correct_analysis', '')
                iteration_data['incorrect_analysis'] = prompt_gen_data.get('incorrect_analysis', '')
        
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                # If prompt not found in prompt_gen_file, try to get it from eval_file
                if not iteration_data['prompt']:
                    iteration_data['prompt'] = eval_data.get('prompt', '')
                # Add evaluation data
                iteration_data['evaluation_details'] = eval_data.get('evaluations', [])
        
        iterations.append(iteration_data)

    # Generate HTML content
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
    with open(os.path.join(log_dir, 'combined_dashboard_multiclass.html'), 'w') as f:
        f.write(html_content)
