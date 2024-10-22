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
    # Remove pairs where either y_true or y_pred is NaN
    valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = np.array(y_true)[valid_indices]
    y_pred_valid = np.array(y_pred)[valid_indices]
    
    if len(y_true_valid) == 0:
        # If no valid pairs, return a placeholder image
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        plt.title('Confusion Matrix (No Valid Data)')
        plt.axis('off')
    else:
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    # Save the plot to a base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

import numpy as np

def generate_confusion_matrix(y_true, y_pred):
    # Remove pairs where either y_true or y_pred is NaN
    valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = np.array(y_true)[valid_indices]
    y_pred_valid = np.array(y_pred)[valid_indices]
    
    if len(y_true_valid) == 0:
        # If no valid pairs, return a placeholder image
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        plt.title('Confusion Matrix (No Valid Data)')
        plt.axis('off')
    else:
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    # Save the plot to a base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_iteration_dashboard(log_dir: str, iteration: int, results: dict, current_prompt: str, output_format_prompt: str, initial_prompt: str):
    """Generate and save an HTML dashboard for a single iteration."""
    
    # Load evaluation results
    evaluation_file = os.path.join(log_dir, f'iteration_{iteration}_evaluation.json')
    with open(evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Convert evaluation results to DataFrame
    df = pd.DataFrame(evaluation_data['evaluations'])
    
    # Try to normalize the raw_output column
    def parse_raw_output(raw_output):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            try:
                return literal_eval(raw_output)
            except:
                return raw_output

    df['parsed_output'] = df['raw_output'].apply(parse_raw_output)
    
    # Try to normalize the parsed output
    try:
        normalized = pd.json_normalize(df['parsed_output'])
        # Remove any columns that are already in the main DataFrame
        normalized = normalized[[col for col in normalized.columns if col not in df.columns]]
        # Combine the original DataFrame with the normalized data
        df = pd.concat([df, normalized], axis=1)
    except:
        # If normalization fails, we'll just use the original DataFrame
        pass
    
    # Reorder columns
    ordered_columns = ['text', 'label', 'transformed_output', 'result'] + [col for col in df.columns if col not in ['text', 'label', 'transformed_output', 'result']]
    df = df[ordered_columns]
    
    # Convert DataFrame back to list of dicts for Jinja template
    evaluation_results = df.to_dict('records')
    
    # Get column names for the table header
    table_headers = df.columns.tolist()
    
    # Generate confusion matrix
    y_true = df['label'].tolist()
    y_pred = df['transformed_output'].tolist()
    
    # Add debugging information
    print(f"Number of samples: {len(y_true)}")
    print(f"Number of NaN in y_true: {sum(np.isnan(y_true))}")
    print(f"Number of NaN in y_pred: {sum(np.isnan(y_pred))}")
    print(f"Sample of y_pred: {y_pred[:10]}")  # Print first 10 predictions
    
    confusion_matrix_image = generate_confusion_matrix(y_true, y_pred)
    
    template = Template('''
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
        <div class="container">
            <h1>Iteration {{ iteration }} Dashboard</h1>
            
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
            
            <h2>Current Prompt</h2>
            <pre class="prompt">{{ current_prompt }}</pre>
            
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
        </div>
        
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
            });
        </script>
    </body>
    </html>
    ''')
    
    html_content = template.render(
        iteration=iteration,
        results=results,
        current_prompt=current_prompt,
        evaluation_results=evaluation_results,
        table_headers=table_headers,
        confusion_matrix_image=confusion_matrix_image
    )
    
    with open(os.path.join(log_dir, f'iteration_{iteration}_dashboard.html'), 'w') as f:
        f.write(html_content)

def generate_combined_dashboard(log_dir: str, all_metrics: list, best_prompt: str, output_format_prompt: str):
    """Generate and save a combined HTML dashboard for all iterations."""
    # Find the best metrics (highest F1 score)
    best_metrics = max(all_metrics, key=lambda x: x['f1'])
    best_iteration = best_metrics['iteration']
    
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Combined Experiment Dashboard</title>
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
            <h1>Combined Experiment Dashboard</h1>
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
                    <td{% if metrics.precision == max_values.precision %} class="best-value"{% endif %}>{{ "%.4f"|format(metrics.precision) }}</td>
                    <td{% if metrics.recall == max_values.recall %} class="best-value"{% endif %}>{{ "%.4f"|format(metrics.recall) }}</td>
                    <td{% if metrics.accuracy == max_values.accuracy %} class="best-value"{% endif %}>{{ "%.4f"|format(metrics.accuracy) }}</td>
                    <td{% if metrics.f1 == max_values.f1 %} class="best-value"{% endif %}>{{ "%.4f"|format(metrics.f1) }}</td>
                    <td{% if metrics.valid_predictions == max_values.valid_predictions %} class="best-value"{% endif %}>{{ metrics.valid_predictions }}</td>
                    <td{% if metrics.invalid_predictions == min_values.invalid_predictions %} class="best-value"{% endif %}>{{ metrics.invalid_predictions }}</td>
                </tr>
                {% endfor %}
            </table>
            <div id="metricsChart"></div>
            <div id="validityChart"></div>
            <h2>Best Prompt - Iteration {{ best_iteration }}, F1 Score: {{ best_metrics.f1|round(4) }}</h2>
            <pre class="prompt">{{ best_prompt }}</pre>
            <h2>Best Metrics Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Precision</td><td>{{ best_metrics.precision|round(4) }}</td></tr>
                <tr><td>Recall</td><td>{{ best_metrics.recall|round(4) }}</td></tr>
                <tr><td>Accuracy</td><td>{{ best_metrics.accuracy|round(4) }}</td></tr>
                <tr><td>F1 Score</td><td>{{ best_metrics.f1|round(4) }}</td></tr>
                <tr><td>Valid Predictions</td><td>{{ best_metrics.valid_predictions }}</td></tr>
                <tr><td>Invalid Predictions</td><td>{{ best_metrics.invalid_predictions }}</td></tr>
            </table>
            <h2>Iterations</h2>
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
                <h4>Prompt</h4>
                <pre class="prompt">{{ iteration.prompt }}</pre>
                <div class="analysis">
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
    
    # Calculate max values for highlighting
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
    
    with open(os.path.join(log_dir, 'combined_dashboard.html'), 'w') as f:
        f.write(html_content)
