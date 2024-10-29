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

# Add this at the top with other imports
from jinja2 import Template

# Add these template definitions at the top of the file, after the imports

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iteration {{ iteration }} Dashboard</title>
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
        .chart-container { display: flex; justify-content: space-between; margin-bottom: 20px; }
        .chart { width: 48%; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; table-layout: fixed; }
        th, td { 
            width: 150px; 
            height: 50px; 
            overflow: hidden; 
            text-overflow: ellipsis; 
            white-space: nowrap; 
            border: 1px solid #ddd; padding: 8px; text-align: left; 
        }
        th { background-color: #f2f2f2; }
        .truncate {
            max-width: 200px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
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
        }
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
            cursor: pointer;
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
    </style>
</head>
<body>
    <h1>Iteration {{ iteration }} Dashboard</h1>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <h3>Precision</h3>
                <p>{{ "%.4f"|format(results.precision) }}</p>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <p>{{ "%.4f"|format(results.recall) }}</p>
            </div>
            <div class="metric-card">
                <h3>Accuracy</h3>
                <p>{{ "%.4f"|format(results.accuracy) }}</p>
            </div>
            <div class="metric-card">
                <h3>F1 Score</h3>
                <p>{{ "%.4f"|format(results.f1) }}</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Confusion Matrix</h2>
        <div class="confusion-matrix">
            <img src="data:image/png;base64,{{ confusion_matrix_image }}" alt="Confusion Matrix">
        </div>
    </div>

    <div class="section">
        <h2>Current Prompt</h2>
        <div class="prompt-box">{{ current_prompt }}</div>
    </div>

    <div class="section">
        <h2>Evaluation Results</h2>
        <table>
            <thead>
                <tr>
                    {% for header in table_headers %}
                    <th>{{ header }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for result in evaluation_results %}
                <tr>
                    {% for header in table_headers %}
                    <td>{{ result[header] }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <h2>Analyses and Prompts</h2>
    
    <h3>False Positives Analysis</h3>
    <h4>Prompt Used:</h4>
    <pre class="prompt">{{ results.prompts_used.fp_prompt }}</pre>
    <h4>Analysis:</h4>
    <pre class="analysis">{{ results.fp_analysis }}</pre>
    
    <h3>False Negatives Analysis</h3>
    <h4>Prompt Used:</h4>
    <pre class="prompt">{{ results.prompts_used.fn_prompt }}</pre>
    <h4>Analysis:</h4>
    <pre class="analysis">{{ results.fn_analysis }}</pre>
    
    <h3>True Positives Analysis</h3>
    <h4>Prompt Used:</h4>
    <pre class="prompt">{{ results.prompts_used.tp_prompt }}</pre>
    <h4>Analysis:</h4>
    <pre class="analysis">{{ results.tp_analysis }}</pre>
    
    <h3>Invalid Outputs Analysis</h3>
    <h4>Prompt Used:</h4>
    <pre class="prompt">{{ results.prompts_used.invalid_prompt }}</pre>
    <h4>Analysis:</h4>
    <pre class="analysis">{{ results.invalid_analysis }}</pre>
    
    <h3>Prompt Engineer Input</h3>
    <pre class="prompt">{{ results.prompts_used.prompt_engineer_input }}</pre>
    
    <h3>New Generated Prompt</h3>
    <pre class="prompt">{{ results.new_prompt }}</pre>
    
    <h3>Validation and Improvement</h3>
    <h4>Validation Result:</h4>
    <pre class="analysis">{{ results.validation_result if results.validation_result != "Skipped" else "Validation was skipped for this iteration." }}</pre>
    
    <h3>Final Improved Prompt for Next Iteration</h3>
    <pre class="prompt">{{ results.improved_prompt }}</pre>

    <div id="myModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <pre id="modalText"></pre>
        </div>
    </div>
    
    <script>
        $(document).ready(function() {
            $('#resultsTable').DataTable({
                pageLength: 10,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                order: [[4, 'desc'], [5, 'desc']],
                columnDefs: [
                    {
                        targets: '_all',
                        render: function(data, type, row) {
                            if (type === 'display') {
                                return '<div title="' + data + '">' + data + '</div>';
                            }
                            return data;
                        }
                    }
                ]
            });
        });

        // Modal functionality
        var modal = document.getElementById("myModal");
        var span = document.getElementsByClassName("close")[0];

        $('#resultsTable').on('dblclick', 'td', function() {
            var cellContent = $(this).text();
            $('#modalText').text(cellContent);
            modal.style.display = "block";
        });

        span.onclick = function() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    </script>
</body>
</html>
'''

COMBINED_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Experiment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
        }
        .prompt { 
            white-space: pre-wrap; 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px;
            border: 1px solid #e9ecef;
            font-family: monospace;
        }
        .iteration { 
            margin-bottom: 40px; 
            border: 1px solid #e9ecef; 
            padding: 20px; 
            border-radius: 8px;
            background: white;
        }
        .metrics { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 20px;
            margin: 20px 0;
        }
        .metric { 
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-width: 200px;
            text-align: center;
        }
        .metric strong {
            display: block;
            color: #666;
            margin-bottom: 5px;
        }
        .analysis { 
            margin-top: 20px;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-bottom: 20px;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td { 
            border: 1px solid #e9ecef; 
            padding: 12px; 
            text-align: right;
        }
        th { 
            background-color: #f8f9fa;
            color: #495057;
            font-weight: 600;
        }
        .best-value { 
            font-weight: bold; 
            color: #28a745;
            background-color: #e8f5e9;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.2em; }
        .chart-container {
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Combined Experiment Dashboard</h1>

        <div class="section">
            <h2>Performance Overview</h2>
            <div class="chart-container">
                <div id="metricsChart"></div>
            </div>
            <div class="chart-container">
                <div id="validityChart"></div>
            </div>
        </div>

        <div class="section">
            <h2>Summary of All Iterations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Iteration</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Accuracy</th>
                        <th>F1-score</th>
                        <th>Valid Predictions</th>
                        <th>Invalid Predictions</th>
                    </tr>
                </thead>
                <tbody>
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
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Best Performing Prompt (Iteration {{ best_iteration }})</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>F1 Score</strong>
                    {{ "%.4f"|format(best_metrics.f1) }}
                </div>
                <div class="metric">
                    <strong>Precision</strong>
                    {{ "%.4f"|format(best_metrics.precision) }}
                </div>
                <div class="metric">
                    <strong>Recall</strong>
                    {{ "%.4f"|format(best_metrics.recall) }}
                </div>
                <div class="metric">
                    <strong>Accuracy</strong>
                    {{ "%.4f"|format(best_metrics.accuracy) }}
                </div>
            </div>
            <pre class="prompt">{{ best_prompt }}</pre>
            
            <h3>Output Format</h3>
            <pre class="prompt">{{ output_format_prompt }}</pre>
        </div>

        <div class="section">
            <h2>Detailed Iteration History</h2>
            {% for iteration in iterations %}
            <div class="iteration">
                <h3>Iteration {{ iteration.number }}</h3>
                <div class="metrics">
                    <div class="metric">
                        <strong>Precision</strong>
                        {{ "%.4f"|format(iteration.precision) }}
                    </div>
                    <div class="metric">
                        <strong>Recall</strong>
                        {{ "%.4f"|format(iteration.recall) }}
                    </div>
                    <div class="metric">
                        <strong>Accuracy</strong>
                        {{ "%.4f"|format(iteration.accuracy) }}
                    </div>
                    <div class="metric">
                        <strong>F1 Score</strong>
                        {{ "%.4f"|format(iteration.f1) }}
                    </div>
                </div>
                
                <h4>Prompt Used</h4>
                <pre class="prompt">{{ iteration.prompt }}</pre>
                
                <div class="analysis">
                    <h4>Analysis Results</h4>
                    <div class="metrics">
                        <div class="metric">
                            <strong>Valid Predictions</strong>
                            {{ iteration.valid_predictions }}
                        </div>
                        <div class="metric">
                            <strong>Invalid Predictions</strong>
                            {{ iteration.invalid_predictions }}
                        </div>
                    </div>
                    
                    <h4>False Positives Analysis</h4>
                    <pre class="prompt">{{ iteration.fp_analysis }}</pre>
                    
                    <h4>False Negatives Analysis</h4>
                    <pre class="prompt">{{ iteration.fn_analysis }}</pre>
                    
                    <h4>True Positives Analysis</h4>
                    <pre class="prompt">{{ iteration.tp_analysis }}</pre>
                    
                    <h4>Invalid Outputs Analysis</h4>
                    <pre class="prompt">{{ iteration.invalid_analysis }}</pre>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        var metricsData = {{ metrics_data|tojson }};
        var metricsLayout = {
            title: 'Metrics Across Iterations',
            xaxis: {title: 'Iteration'},
            yaxis: {title: 'Score', range: [0, 1]},
            template: 'plotly_white'
        };
        Plotly.newPlot('metricsChart', metricsData, metricsLayout);

        var validityData = {{ validity_data|tojson }};
        var validityLayout = {
            title: 'Valid vs Invalid Predictions Across Iterations',
            xaxis: {title: 'Iteration'},
            yaxis: {title: 'Number of Predictions'},
            barmode: 'stack',
            template: 'plotly_white'
        };
        Plotly.newPlot('validityChart', validityData, validityLayout);
    </script>
</body>
</html>
'''

def generate_confusion_matrix(y_true, y_pred):
    """Generate a confusion matrix visualization for binary classification."""
    # Add debugging information
    print(f"Number of samples: {len(y_true)}")
    print(f"Number of NaN in y_true: {sum(np.isnan(y_true))}")
    print(f"Number of NaN in y_pred: {sum(np.isnan(y_pred))}")
    print(f"Sample of y_pred: {y_pred[:10]}")  # Print first 10 predictions
    
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
    
    # Create template and render HTML
    template = Template(DASHBOARD_TEMPLATE)
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
    
    # Create template and render HTML
    template = Template(COMBINED_DASHBOARD_TEMPLATE)
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

