"""
Templates for dashboard generation.
Contains HTML templates for both binary and multiclass classification dashboards.
"""

from jinja2 import Template

# Common minimalistic styles shared between templates
COMMON_STYLES = '''
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto;
            padding: 2rem;
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .metric { 
            text-align: center; 
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .metric strong {
            display: block;
            color: #34495e;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        .metric span {
            font-size: 1.5rem;
            font-weight: 600;
            color: #3498db;
        }
        .prompt, .analysis { 
            background-color: #fff; 
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
            border: 1px solid #eee;
        }
        .confusion-matrix { 
            width: 50%; 
            margin: 20px auto; 
            text-align: center; 
            background: white;
            padding: 1rem;
            border-radius: 8px;
        }
        .confusion-matrix img { 
            max-width: 100%; 
            height: auto; 
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
            border-bottom: 1px solid #eee;
            text-align: left;
        }
        th { 
            background-color: #f8f9fa;
            font-weight: 600;
            color: #34495e;
        }
        .section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        .close:hover {
            color: #666;
        }
        .chart-container {
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }
        .best-value {
            color: #2ecc71;
            font-weight: 600;
        }
        .dataTables_wrapper {
            padding: 1rem;
            background: white;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .dataTables_filter input,
        .dataTables_length select {
            padding: 0.5rem;
            border: 1px solid #eee;
            border-radius: 4px;
            margin: 0 0.5rem;
        }
        h1, h2, h3 {
            color: #34495e;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .result-icon {
            font-weight: bold;
        }
        .result-success {
            color: #2ecc71;
        }
        .result-failure {
            color: #e74c3c;
        }
    </style>
'''

# Template for both binary and multiclass iteration dashboards
ITERATION_TEMPLATE = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Dashboard - Iteration {{ iteration }}</title>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.css">
    <script type="text/javascript" src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    ''' + COMMON_STYLES + '''
</head>
<body>
    <div class="container">
        <h1>Iteration {{ iteration }} Dashboard</h1>
        
        <div class="section">
            <h2>Evaluation Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Precision</strong> 
                    <span>{{ "%.4f"|format(results.precision) }}</span>
                </div>
                <div class="metric">
                    <strong>Recall</strong> 
                    <span>{{ "%.4f"|format(results.recall) }}</span>
                </div>
                <div class="metric">
                    <strong>Accuracy</strong> 
                    <span>{{ "%.4f"|format(results.accuracy) }}</span>
                </div>
                <div class="metric">
                    <strong>F1 Score</strong> 
                    <span>{{ "%.4f"|format(results.f1) }}</span>
                </div>
                <div class="metric">
                    <strong>Valid Predictions</strong> 
                    <span>{{ results.valid_predictions }}</span>
                </div>
                <div class="metric">
                    <strong>Invalid Predictions</strong> 
                    <span>{{ results.invalid_predictions }}</span>
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
            <pre class="prompt">{{ current_prompt }}</pre>
        </div>
        
        <div class="section">
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
        </div>

        <div id="myModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <pre id="modalText"></pre>
            </div>
        </div>

        <div class="section">
            <h2>Analysis</h2>
            {% if results.correct_analysis %}
            <h3>Correct Predictions Analysis</h3>
            <pre class="analysis">{{ results.correct_analysis }}</pre>
            {% endif %}
            
            {% if results.incorrect_analysis %}
            <h3>Incorrect Predictions Analysis</h3>
            <pre class="analysis">{{ results.incorrect_analysis }}</pre>
            {% endif %}

            {% if results.fp_analysis %}
            <h3>False Positives Analysis</h3>
            <pre class="analysis">{{ results.fp_analysis }}</pre>
            {% endif %}

            {% if results.fn_analysis %}
            <h3>False Negatives Analysis</h3>
            <pre class="analysis">{{ results.fn_analysis }}</pre>
            {% endif %}

            {% if results.tp_analysis %}
            <h3>True Positives Analysis</h3>
            <pre class="analysis">{{ results.tp_analysis }}</pre>
            {% endif %}

            {% if results.invalid_analysis %}
            <h3>Invalid Outputs Analysis</h3>
            <pre class="analysis">{{ results.invalid_analysis }}</pre>
            {% endif %}
        </div>
    </div>

    <script>
        $(document).ready(function() {
            // Initialize DataTable
            var table = $('#resultsTable').DataTable({
                pageLength: 10,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                order: [[0, 'asc']]
            });

            // Modal functionality
            var modal = document.getElementById("myModal");
            var modalText = document.getElementById("modalText");
            var span = document.getElementsByClassName("close")[0];

            $('#resultsTable tbody').on('click', 'td', function() {
                var cellContent = table.cell(this).data();
                if (cellContent) {
                    modalText.textContent = cellContent;
                    modal.style.display = "block";
                }
            });

            span.onclick = function() {
                modal.style.display = "none";
            }

            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }

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

# Template for combined dashboard (works for both binary and multiclass)
COMBINED_TEMPLATE = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Classification Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    ''' + COMMON_STYLES + '''
</head>
<body>
    <div class="container">
        <h1>Combined Classification Dashboard</h1>
        
        <div class="section">
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
        </div>

        <div class="section">
            <h2>Performance Metrics Over Time</h2>
            <div id="metricsChart" class="chart-container"></div>
            <div id="validityChart" class="chart-container"></div>
        </div>

        <div class="section">
            <h2>Best Performing Prompt (Iteration {{ best_iteration }})</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>F1 Score</strong>
                    <span>{{ "%.4f"|format(best_metrics.f1) }}</span>
                </div>
                <div class="metric">
                    <strong>Precision</strong>
                    <span>{{ "%.4f"|format(best_metrics.precision) }}</span>
                </div>
                <div class="metric">
                    <strong>Recall</strong>
                    <span>{{ "%.4f"|format(best_metrics.recall) }}</span>
                </div>
                <div class="metric">
                    <strong>Accuracy</strong>
                    <span>{{ "%.4f"|format(best_metrics.accuracy) }}</span>
                </div>
            </div>
            <pre class="prompt">{{ best_prompt }}</pre>
        </div>

        <div class="section">
            <h2>Detailed Iteration History</h2>
            {% for iteration in iterations %}
            <div class="section">
                <h3>Iteration {{ iteration.number }}</h3>
                <div class="metrics">
                    <div class="metric">
                        <strong>Precision</strong>
                        <span>{{ "%.4f"|format(iteration.precision) }}</span>
                    </div>
                    <div class="metric">
                        <strong>Recall</strong>
                        <span>{{ "%.4f"|format(iteration.recall) }}</span>
                    </div>
                    <div class="metric">
                        <strong>Accuracy</strong>
                        <span>{{ "%.4f"|format(iteration.accuracy) }}</span>
                    </div>
                    <div class="metric">
                        <strong>F1 Score</strong>
                        <span>{{ "%.4f"|format(iteration.f1) }}</span>
                    </div>
                </div>
                
                <h4>Prompt Used</h4>
                <pre class="prompt">{{ iteration.prompt }}</pre>
                
                <div class="analysis">
                    {% if iteration.correct_analysis %}
                    <h4>Correct Predictions Analysis</h4>
                    <pre class="analysis">{{ iteration.correct_analysis }}</pre>
                    {% endif %}
                    
                    {% if iteration.incorrect_analysis %}
                    <h4>Incorrect Predictions Analysis</h4>
                    <pre class="analysis">{{ iteration.incorrect_analysis }}</pre>
                    {% endif %}

                    {% if iteration.fp_analysis %}
                    <h4>False Positives Analysis</h4>
                    <pre class="analysis">{{ iteration.fp_analysis }}</pre>
                    {% endif %}

                    {% if iteration.fn_analysis %}
                    <h4>False Negatives Analysis</h4>
                    <pre class="analysis">{{ iteration.fn_analysis }}</pre>
                    {% endif %}

                    {% if iteration.tp_analysis %}
                    <h4>True Positives Analysis</h4>
                    <pre class="analysis">{{ iteration.tp_analysis }}</pre>
                    {% endif %}

                    {% if iteration.invalid_analysis %}
                    <h4>Invalid Outputs Analysis</h4>
                    <pre class="analysis">{{ iteration.invalid_analysis }}</pre>
                    {% endif %}
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
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };
        Plotly.newPlot('metricsChart', metricsData, metricsLayout);

        var validityData = {{ validity_data|tojson }};
        var validityLayout = {
            title: 'Valid vs Invalid Predictions',
            xaxis: {title: 'Iteration'},
            yaxis: {title: 'Number of Predictions'},
            barmode: 'stack',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };
        Plotly.newPlot('validityChart', validityData, validityLayout);
    </script>
</body>
</html>
''')

# Use the same template for both binary and multiclass
BINARY_ITERATION_TEMPLATE = ITERATION_TEMPLATE
MULTICLASS_ITERATION_TEMPLATE = ITERATION_TEMPLATE
BINARY_COMBINED_TEMPLATE = COMBINED_TEMPLATE
MULTICLASS_COMBINED_TEMPLATE = COMBINED_TEMPLATE 