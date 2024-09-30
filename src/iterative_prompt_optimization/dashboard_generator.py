import os
import json
from jinja2 import Template
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def generate_iteration_dashboard(log_dir: str, iteration: int, results: dict, current_prompt: str, output_format_prompt: str):
    """Generate and save an HTML dashboard for a single iteration."""
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iteration {{ iteration }} Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .metrics { display: flex; justify-content: space-around; margin-bottom: 20px; }
            .metric { text-align: center; }
            .prompt { white-space: pre-wrap; background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
            .chart-container { display: flex; justify-content: space-between; margin-bottom: 20px; }
            .chart { width: 48%; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Iteration {{ iteration }} Dashboard</h1>
            <div class="metrics">
                <div class="metric">
                    <h3>Precision</h3>
                    <p>{{ "%.4f"|format(results.precision) }}</p>
                </div>
                <div class="metric">
                    <h3>Recall</h3>
                    <p>{{ "%.4f"|format(results.recall) }}</p>
                </div>
                <div class="metric">
                    <h3>Accuracy</h3>
                    <p>{{ "%.4f"|format(results.accuracy) }}</p>
                </div>
                <div class="metric">
                    <h3>F1 Score</h3>
                    <p>{{ "%.4f"|format(results.f1) }}</p>
                </div>
                <div class="metric">
                    <h3>Valid Predictions</h3>
                    <p>{{ results.valid_predictions }}</p>
                </div>
                <div class="metric">
                    <h3>Invalid Predictions</h3>
                    <p>{{ results.invalid_predictions }}</p>
                </div>
            </div>
            <div class="chart-container">
                <div id="metricsChart" class="chart"></div>
                <div id="validityChart" class="chart"></div>
            </div>
            <h2>Prompt</h2>
            <pre class="prompt">{{ current_prompt }}</pre>
            <h2>Output Format</h2>
            <pre class="prompt">{{ output_format_prompt }}</pre>
            <h2>Analyses</h2>
            <h3>False Positives Analysis</h3>
            <pre class="analysis">{{ results.fp_analysis }}</pre>
            <h3>False Negatives Analysis</h3>
            <pre class="analysis">{{ results.fn_analysis }}</pre>
            <h3>True Positives Analysis</h3>
            <pre class="analysis">{{ results.tp_analysis }}</pre>
            <h3>Invalid Outputs Analysis</h3>
            <pre class="analysis">{{ results.invalid_analysis }}</pre>
        </div>
        <script>
            var metricsData = [
                {
                    x: ['Precision', 'Recall', 'Accuracy', 'F1 Score'],
                    y: [{{ results.precision }}, {{ results.recall }}, {{ results.accuracy }}, {{ results.f1 }}],
                    type: 'bar'
                }
            ];
            var metricsLayout = {
                title: 'Performance Metrics',
                yaxis: {range: [0, 1]}
            };
            Plotly.newPlot('metricsChart', metricsData, metricsLayout);

            var validityData = [
                {
                    values: [{{ results.valid_predictions }}, {{ results.invalid_predictions }}],
                    labels: ['Valid', 'Invalid'],
                    type: 'pie',
                    textinfo: "label+percent",
                    insidetextorientation: "radial"
                }
            ];
            var validityLayout = {
                title: 'Prediction Validity'
            };
            Plotly.newPlot('validityChart', validityData, validityLayout);
        </script>
    </body>
    </html>
    ''')
    
    html_content = template.render(iteration=iteration, results=results, current_prompt=current_prompt, output_format_prompt=output_format_prompt)
    
    with open(os.path.join(log_dir, f'iteration_{iteration}_dashboard.html'), 'w') as f:
        f.write(html_content)

def generate_experiment_dashboard(log_dir: str, all_metrics: list, best_prompt: str, output_format_prompt: str):
    """Generate and save an HTML dashboard for the entire experiment."""
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Experiment Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .prompt { white-space: pre-wrap; background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Experiment Dashboard</h1>
            <div id="metricsChart"></div>
            <h2>Best Prompt</h2>
            <pre class="prompt">{{ best_prompt }}</pre>
            <h2>Output Format</h2>
            <pre class="prompt">{{ output_format_prompt }}</pre>
        </div>
        <script>
            var data = {{ plotly_data|tojson }};
            var layout = {
                title: 'Metrics Across Iterations',
                xaxis: {title: 'Iteration'},
                yaxis: {title: 'Score', range: [0, 1]}
            };
            Plotly.newPlot('metricsChart', data, layout);
        </script>
    </body>
    </html>
    ''')
    
    # Prepare data for Plotly
    metrics = ['precision', 'recall', 'accuracy', 'f1', 'valid_predictions', 'invalid_predictions']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    plotly_data = []
    
    for i, metric in enumerate(metrics):
        plotly_data.append({
            'x': [m['iteration'] for m in all_metrics],
            'y': [m[metric] for m in all_metrics],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': metric.capitalize().replace('_', ' '),
            'line': {'color': colors[i]}
        })
    
    html_content = template.render(plotly_data=plotly_data, best_prompt=best_prompt, output_format_prompt=output_format_prompt)
    
    with open(os.path.join(log_dir, 'experiment_dashboard.html'), 'w') as f:
        f.write(html_content)

def generate_combined_dashboard(log_dir: str, all_metrics: list, best_prompt: str, output_format_prompt: str):
    """Generate and save a combined HTML dashboard for all iterations."""
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Combined Experiment Dashboard</h1>
            <div id="metricsChart"></div>
            <div id="validityChart"></div>
            <h2>Best Prompt - F1 Score: {{ best_metrics.f1|round(4) }}</h2>
            <pre class="prompt">{{ best_prompt }}</pre>
            <h2>Output Format</h2>
            <pre class="prompt">{{ output_format_prompt }}</pre>
            <h2>Summary of Best Metrics</h2>
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
    
    # Get the best metrics (assuming the last iteration has the best F1 score)
    best_metrics = all_metrics[-1]
    
    html_content = template.render(
        metrics_data=metrics_data,
        validity_data=validity_data,
        best_prompt=best_prompt,
        output_format_prompt=output_format_prompt,
        iterations=iterations,
        best_metrics=best_metrics
    )
    
    with open(os.path.join(log_dir, 'combined_dashboard.html'), 'w') as f:
        f.write(html_content)