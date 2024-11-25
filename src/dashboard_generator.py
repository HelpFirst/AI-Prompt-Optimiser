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

# Common styles to be included in both templates
COMMON_STYLES = '''
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --background-color: #f8f9fa;
            --prompt-bg-color: #fff9e6;
            --success-color: #2ecc71;
            --border-color: #e9ecef;
        }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--primary-color);
            line-height: 1.6;
        }
        
        .header {
            background: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            height: 40px;
            margin-right: 1rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .metric {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        
        .metric:hover {
            transform: translateY(-2px);
        }
        
        .metric strong {
            display: block;
            color: var(--secondary-color);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        
        .metric span {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-color);
        }
        
        .prompt, .analysis {
            background-color: var(--prompt-bg-color);
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 1.5rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }
        
        th, td {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--background-color);
            font-weight: 600;
            text-align: left;
            color: var(--secondary-color);
        }
        
        .best-value {
            color: var(--success-color);
            font-weight: 600;
        }
        
        /* Collapsible sections */
        .collapsible {
            background: none;
            border: none;
            padding: 1rem;
            width: 100%;
            text-align: left;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        .collapsible:after {
            content: '\\002B';
            font-weight: bold;
            float: right;
            margin-left: 5px;
        }
        
        .active:after {
            content: '\\2212';
        }
        
        .content {
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background-color: white;
        }
        
        .content.show {
            max-height: none;  /* Changed from fixed height to allow any height */
        }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Add styles for text cells */
        .truncated-text {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            padding: 4px;
        }
        
        .truncated-text:hover {
            background-color: var(--prompt-bg-color);
            border-radius: 4px;
        }
        
        /* Modal styling */
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
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-text {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
            padding: 1rem;
            background-color: var(--prompt-bg-color);
            border-radius: 4px;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: var(--accent-color);
        }
        
        /* Table and cell styles */
        .truncate-cell {
            max-width: 300px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .truncated-text {
            max-width: 300px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            cursor: pointer;
        }
        
        .truncated-text:hover {
            color: var(--accent-color);
            text-decoration: underline;
        }
        
        /* Modal styles */
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
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: var(--prompt-bg-color);
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: var(--accent-color);
        }
        
        /* DataTables specific styling */
        .dataTables_wrapper {
            padding: 1rem;
            background: white;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .dataTables_filter input {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            margin-left: 0.5rem;
        }
        
        .dataTables_length select {
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            margin: 0 0.5rem;
        }
    </style>
'''

# Add collapsible functionality JavaScript
COLLAPSIBLE_SCRIPT = '''
    <script>
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                    
                    // If this content contains other collapsibles, adjust parent's maxHeight
                    var parent = content.parentElement;
                    while (parent && parent.classList.contains("content")) {
                        parent.style.maxHeight = parent.scrollHeight + "px";
                        parent = parent.parentElement;
                    }
                }
            });
        }
    </script>
'''

# Add this function definition before the templates
def get_base64_logo():
    """Get the HelpFirst logo as a base64 encoded string."""
    import os
    import base64
    
    # Get the path to the logo relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, 'assets', 'helpfirst-logo.jpg')
    
    try:
        with open(logo_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Warning: Logo file not found at {logo_path}")
        return ""

# Now we can use the function
LOGO_BASE64 = get_base64_logo()

# Update the logo in the template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iteration {{ iteration }} Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Fira+Code&display=swap">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.24/css/jquery.dataTables.css">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.7.0/css/buttons.dataTables.min.css">
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/buttons/1.7.0/js/dataTables.buttons.min.js"></script>
    ''' + COMMON_STYLES + '''
    <style>
        /* DataTables specific styling */
        .dataTables_wrapper {
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        table.dataTable {
            width: 100% !important;
            margin: 0 !important;
        }
        
        .dataTables_filter input {
            padding: 5px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }
        
        /* Modal styling */
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
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 8px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <img src="data:image/jpeg;base64,''' + LOGO_BASE64 + '''" alt="HelpFirst Logo" class="logo">
        <h1>Iteration {{ iteration }} Dashboard</h1>
    </div>
    
    <div class="container">
        <button class="collapsible">Performance Metrics</button>
        <div class="content section">
            <div class="metrics">
                <div class="metric">
                    <strong>Precision</strong>
                    <span>{{ "%.3f"|format(results.precision) }}</span>
                </div>
                <div class="metric">
                    <strong>Recall</strong>
                    <span>{{ "%.3f"|format(results.recall) }}</span>
                </div>
                <div class="metric">
                    <strong>F1 Score</strong>
                    <span>{{ "%.3f"|format(results.f1) }}</span>
                </div>
                <div class="metric">
                    <strong>Accuracy</strong>
                    <span>{{ "%.3f"|format(results.accuracy) }}</span>
                </div>
            </div>
        </div>

        <button class="collapsible">Confusion Matrix</button>
        <div class="content section">
            <img src="data:image/png;base64,{{ confusion_matrix_image }}" alt="Confusion Matrix">
        </div>

        <button class="collapsible">Current Prompt</button>
        <div class="content section">
            <div class="prompt">{{ current_prompt }}</div>
        </div>

        <div class="section">
            <h2>Detailed Results</h2>
            <table id="resultsTable" class="display">
                <thead>
                    <tr>
                        <th>Text</th>
                        <th>True Label</th>
                        <th>Prediction</th>
                        <th>Result</th>
                        <th>Correct?</th>
                        <th>Valid?</th>
                        <th>Chain of Thought</th>
                        <th>Raw Output</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in evaluation_results %}
                    <tr>
                        <td>
                            <div class="truncated-text" data-full-text="{{ result.text|replace('`', '\\`')|replace('"', '\\"')|replace('\n', '\\n')|replace("'", "\\'") }}">
                                {{ result.text|truncate(100) }}
                            </div>
                        </td>
                        <td>{{ result.label }}</td>
                        <td>{{ result.transformed_output }}</td>
                        <td>{{ "✅ (TP)" if result.is_correct and result.label == 1 else 
                               "✅ (TN)" if result.is_correct and result.label == 0 else 
                               "❌ (FP)" if not result.is_correct and result.transformed_output == 1 else 
                               "❌ (FN)" if not result.is_correct and result.transformed_output == 0 else 
                               "❌ (Invalid)" if not result.is_valid else "" }}</td>
                        <td>{{ "✅" if result.is_correct else "❌" }}</td>
                        <td>{{ "✅" if result.is_valid else "❌" }}</td>
                        <td>
                            <div class="truncated-text" data-full-text="{{ result.chain_of_thought|replace('`', '\\`')|replace('"', '\\"')|replace('\n', '\\n')|replace("'", "\\'") }}">
                                {{ result.chain_of_thought|truncate(100) }}
                            </div>
                        </td>
                        <td>
                            <div class="truncated-text" data-full-text="{{ result.raw_output|replace('`', '\\`')|replace('"', '\\"')|replace('\n', '\\n')|replace("'", "\\'") }}">
                                {{ result.raw_output|truncate(100) }}
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Modal -->
    <div id="modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2 id="modalTitle"></h2>
            <pre id="modalContent"></pre>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            // Initialize DataTable
            var table = $('#resultsTable').DataTable({
                pageLength: 10,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                order: [],
                scrollX: true
            });

            // Add click handler for truncated text
            $('.truncated-text').on('click', function() {
                var fullText = $(this).data('full-text');
                var title = $(this).closest('tr').find('td:first .truncated-text').data('full-text').substring(0, 100) + '...';
                showModal(title, fullText);
            });
        });

        // Modal functionality
        var modal = document.getElementById("modal");
        var span = document.getElementsByClassName("close")[0];

        function showModal(title, content) {
            content = content.replace(/\\n/g, '\n')
                           .replace(/\\'/g, "'")
                           .replace(/\\"/g, '"')
                           .replace(/\\`/g, '`');
            document.getElementById("modalTitle").textContent = title;
            document.getElementById("modalContent").textContent = content;
            modal.style.display = "block";
        }

        span.onclick = function() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    </script>
    
    ''' + COLLAPSIBLE_SCRIPT + '''
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
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Fira+Code&display=swap">
    ''' + COMMON_STYLES + '''
</head>
<body>
    <div class="header">
        <img src="data:image/jpeg;base64,''' + LOGO_BASE64 + '''" alt="HelpFirst Logo" class="logo">
        <h1>Combined Experiment Dashboard</h1>
    </div>
    
    <div class="container">
        <button class="collapsible">Summary of All Iterations</button>
        <div class="content section">
            <table>
                <thead>
                    <tr>
                        <th>Iteration</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Accuracy</th>
                        <th>Valid Predictions</th>
                        <th>Invalid Predictions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metrics in all_metrics %}
                    <tr>
                        <td>{{ metrics.iteration }}</td>
                        <td {% if metrics.precision == max_values.precision %}class="best-value"{% endif %}>
                            {{ "%.3f"|format(metrics.precision) }}
                        </td>
                        <td {% if metrics.recall == max_values.recall %}class="best-value"{% endif %}>
                            {{ "%.3f"|format(metrics.recall) }}
                        </td>
                        <td {% if metrics.f1 == max_values.f1 %}class="best-value"{% endif %}>
                            {{ "%.3f"|format(metrics.f1) }}
                        </td>
                        <td {% if metrics.accuracy == max_values.accuracy %}class="best-value"{% endif %}>
                            {{ "%.3f"|format(metrics.accuracy) }}
                        </td>
                        <td {% if metrics.valid_predictions == max_values.valid_predictions %}class="best-value"{% endif %}>
                            {{ metrics.valid_predictions }}
                        </td>
                        <td {% if metrics.invalid_predictions == min_values.invalid_predictions %}class="best-value"{% endif %}>
                            {{ metrics.invalid_predictions }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <button class="collapsible">Performance Overview</button>
        <div class="content section">
            <div class="chart-container">
                <div id="metricsChart"></div>
                <div id="validityChart"></div>
            </div>
            <script>
                var metricsData = {{ metrics_data|tojson }};
                var validityData = {{ validity_data|tojson }};
                
                Plotly.newPlot('metricsChart', metricsData, {
                    title: 'Performance Metrics Across Iterations',
                    xaxis: {title: 'Iteration'},
                    yaxis: {title: 'Score'}
                });
                
                Plotly.newPlot('validityChart', validityData, {
                    title: 'Prediction Validity Across Iterations',
                    xaxis: {title: 'Iteration'},
                    yaxis: {title: 'Count'},
                    barmode: 'stack'
                });
            </script>
        </div>
        
        <button class="collapsible">Best Performing Prompt (Iteration {{ best_iteration }})</button>
        <div class="content section">
            <div class="prompt">{{ best_prompt }}</div>
        </div>
        
        <button class="collapsible">Detailed Iteration History</button>
        <div class="content section">
            {% for iteration in iterations %}
            <button class="collapsible">Iteration {{ iteration.number }}</button>
            <div class="content">
                <div class="section">
                    <h3>Metrics</h3>
                    <div class="metrics">
                        <div class="metric">
                            <strong>Precision</strong>
                            <span>{{ "%.3f"|format(iteration.precision) }}</span>
                        </div>
                        <div class="metric">
                            <strong>Recall</strong>
                            <span>{{ "%.3f"|format(iteration.recall) }}</span>
                        </div>
                        <div class="metric">
                            <strong>F1 Score</strong>
                            <span>{{ "%.3f"|format(iteration.f1) }}</span>
                        </div>
                        <div class="metric">
                            <strong>Accuracy</strong>
                            <span>{{ "%.3f"|format(iteration.accuracy) }}</span>
                        </div>
                    </div>
                    
                    <h3>Prompt</h3>
                    <div class="prompt">{{ iteration.prompt }}</div>
                    
                    <h3>Analysis</h3>
                    <div class="analysis">
                        <h4>False Positives Analysis</h4>
                        {{ iteration.fp_analysis }}
                        
                        <h4>False Negatives Analysis</h4>
                        {{ iteration.fn_analysis }}
                        
                        <h4>True Positives Analysis</h4>
                        {{ iteration.tp_analysis }}
                        
                        <h4>Invalid Outputs Analysis</h4>
                        {{ iteration.invalid_analysis }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    ''' + COLLAPSIBLE_SCRIPT + '''
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
    """Generate an HTML dashboard for a single iteration."""
    # Load prompt generation data if available
    prompt_gen_file = os.path.join(log_dir, f'iteration_{iteration}_prompt_generation.json')
    prompt_data = {}
    if os.path.exists(prompt_gen_file):
        with open(prompt_gen_file, 'r') as f:
            prompt_data = json.load(f)
    
    # Convert evaluation results to DataFrame for easier processing
    df = pd.DataFrame(results.get('evaluations', []))
    
    # Ensure all required columns exist with correct names
    required_columns = {
        'text': 'text',
        'label': 'label',
        'transformed_output': 'transformed_output',
        'is_correct': 'is_correct',
        'is_valid': 'is_valid',
        'chain_of_thought': 'chain_of_thought',
        'raw_output': 'raw_output'
    }
    
    for col, default_value in required_columns.items():
        if col not in df.columns:
            df[col] = None
    
    # Convert DataFrame back to list of dictionaries
    evaluation_results = df.to_dict('records')
    
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
        confusion_matrix_image=confusion_matrix_image,
        prompt_data=prompt_data
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
                    'fp_analysis': iteration_json['analysis_results']['false_positives'],
                    'fn_analysis': iteration_json['analysis_results']['false_negatives'],
                    'tp_analysis': iteration_json['analysis_results']['true_positives'],
                    'invalid_analysis': iteration_json['analysis_results']['invalid_outputs'],
                    'prompts_used': iteration_json['prompts_for_analysis']
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

