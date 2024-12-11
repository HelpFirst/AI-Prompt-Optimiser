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
            max-width: 100%;
            word-wrap: break-word;
            line-height: 1.5;
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
            table-layout: fixed;
        }
        th, td { 
            padding: 1rem;
            border-bottom: 1px solid #eee;
            text-align: left;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 300px;
        }
        /* Column widths for evaluation table */
        table#resultsTable th:nth-child(1), 
        table#resultsTable td:nth-child(1) { width: 25%; } /* Text */
        table#resultsTable th:nth-child(2),
        table#resultsTable td:nth-child(2) { width: 10%; } /* True Label */
        table#resultsTable th:nth-child(3),
        table#resultsTable td:nth-child(3) { width: 10%; } /* Predicted */
        table#resultsTable th:nth-child(4),
        table#resultsTable td:nth-child(4) { width: 25%; } /* Chain of Thought */
        table#resultsTable th:nth-child(5),
        table#resultsTable td:nth-child(5) { width: 25%; } /* Raw Output */
        table#resultsTable th:nth-child(6),
        table#resultsTable td:nth-child(6) { width: 5%; } /* Result */
        
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
            overflow-x: auto;
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
        .modal-text {
            white-space: pre-wrap;
            word-wrap: break-word;
            max-width: 100%;
            font-family: monospace;
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
            color: #666;
        }
        .chart-container {
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
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
            overflow-x: auto;
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
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            .metrics {
                grid-template-columns: 1fr;
            }
            .confusion-matrix {
                width: 100%;
            }
            table {
                display: block;
                overflow-x: auto;
            }
            th, td {
                min-width: 120px;
            }
        }
        /* Custom scrollbar for better UX */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        .analysis-section {
            margin-top: 1rem;
        }
        
        .analysis-section h3 {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            color: #34495e;
            font-size: 1.1rem;
        }
        
        .analysis-section h3:first-child {
            margin-top: 0;
        }
        
        .analysis {
            background-color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
            border: 1px solid #eee;
            max-width: 100%;
            word-wrap: break-word;
            line-height: 1.5;
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
                    <span>{{ "%.4f"|format(results.get('precision', 0)) if results.get('precision') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Recall</strong> 
                    <span>{{ "%.4f"|format(results.get('recall', 0)) if results.get('recall') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Accuracy</strong> 
                    <span>{{ "%.4f"|format(results.get('accuracy', 0)) if results.get('accuracy') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>F1 Score</strong> 
                    <span>{{ "%.4f"|format(results.get('f1', 0)) if results.get('f1') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Valid Predictions</strong> 
                    <span>{{ results.get('valid_predictions', 'N/A') }}</span>
                </div>
                <div class="metric">
                    <strong>Invalid Predictions</strong> 
                    <span>{{ results.get('invalid_predictions', 'N/A') }}</span>
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
            <div class="table-responsive">
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
                            <td title="Click to view full content">{{ eval.get(header, 'N/A') }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <div id="myModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <pre class="modal-text" id="modalText"></pre>
            </div>
        </div>

        <div class="section">
            <h2>Analysis</h2>
            <div class="analysis-section">
                {% if results.get('correct_analysis') %}
                <h3>Correct Predictions Analysis</h3>
                <pre class="analysis">{{ results.get('correct_analysis') }}</pre>
                {% endif %}
                
                {% if results.get('incorrect_analysis') %}
                <h3>Incorrect Predictions Analysis</h3>
                <pre class="analysis">{{ results.get('incorrect_analysis') }}</pre>
                {% endif %}

                {% if results.get('fp_analysis') %}
                <h3>False Positives Analysis</h3>
                <pre class="analysis">{{ results.get('fp_analysis') }}</pre>
                {% endif %}

                {% if results.get('fn_analysis') %}
                <h3>False Negatives Analysis</h3>
                <pre class="analysis">{{ results.get('fn_analysis') }}</pre>
                {% endif %}

                {% if results.get('tp_analysis') %}
                <h3>True Positives Analysis</h3>
                <pre class="analysis">{{ results.get('tp_analysis') }}</pre>
                {% endif %}

                {% if results.get('invalid_analysis') %}
                <h3>Invalid Outputs Analysis</h3>
                <pre class="analysis">{{ results.get('invalid_analysis') }}</pre>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            // Initialize DataTable
            var table = $('#resultsTable').DataTable({
                pageLength: 10,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                order: [[0, 'asc']],
                scrollX: true,
                autoWidth: false
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

            $(document).keydown(function(e) {
                if (e.key === "Escape") {
                    modal.style.display = "none";
                }
            });

            // Adjust table column widths on window resize
            $(window).resize(function() {
                table.columns.adjust();
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
            <div class="table-responsive">
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
                        <td>{{ metrics.get('iteration', 'N/A') }}</td>
                        <td{% if metrics.get('precision') == max_values.get('precision') %} class="best-value"{% endif %}>
                            {{ "%.4f"|format(metrics.get('precision', 0)) if metrics.get('precision') is not none else 'N/A' }}
                        </td>
                        <td{% if metrics.get('recall') == max_values.get('recall') %} class="best-value"{% endif %}>
                            {{ "%.4f"|format(metrics.get('recall', 0)) if metrics.get('recall') is not none else 'N/A' }}
                        </td>
                        <td{% if metrics.get('accuracy') == max_values.get('accuracy') %} class="best-value"{% endif %}>
                            {{ "%.4f"|format(metrics.get('accuracy', 0)) if metrics.get('accuracy') is not none else 'N/A' }}
                        </td>
                        <td{% if metrics.get('f1') == max_values.get('f1') %} class="best-value"{% endif %}>
                            {{ "%.4f"|format(metrics.get('f1', 0)) if metrics.get('f1') is not none else 'N/A' }}
                        </td>
                        <td{% if metrics.get('valid_predictions') == max_values.get('valid_predictions') %} class="best-value"{% endif %}>
                            {{ metrics.get('valid_predictions', 'N/A') }}
                        </td>
                        <td{% if metrics.get('invalid_predictions') == min_values.get('invalid_predictions') %} class="best-value"{% endif %}>
                            {{ metrics.get('invalid_predictions', 'N/A') }}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
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
                    <span>{{ "%.4f"|format(best_metrics.get('f1', 0)) if best_metrics.get('f1') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Precision</strong>
                    <span>{{ "%.4f"|format(best_metrics.get('precision', 0)) if best_metrics.get('precision') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Recall</strong>
                    <span>{{ "%.4f"|format(best_metrics.get('recall', 0)) if best_metrics.get('recall') is not none else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <strong>Accuracy</strong>
                    <span>{{ "%.4f"|format(best_metrics.get('accuracy', 0)) if best_metrics.get('accuracy') is not none else 'N/A' }}</span>
                </div>
            </div>
            <pre class="prompt">{{ best_prompt }}</pre>
        </div>

        <div class="section">
            <h2>Detailed Iteration History</h2>
            {% for iteration in iterations %}
            <div class="section">
                <h3>Iteration {{ iteration.get('number', 'N/A') }}</h3>
                <div class="metrics">
                    <div class="metric">
                        <strong>Precision</strong>
                        <span>{{ "%.4f"|format(iteration.get('precision', 0)) if iteration.get('precision') is not none else 'N/A' }}</span>
                    </div>
                    <div class="metric">
                        <strong>Recall</strong>
                        <span>{{ "%.4f"|format(iteration.get('recall', 0)) if iteration.get('recall') is not none else 'N/A' }}</span>
                    </div>
                    <div class="metric">
                        <strong>Accuracy</strong>
                        <span>{{ "%.4f"|format(iteration.get('accuracy', 0)) if iteration.get('accuracy') is not none else 'N/A' }}</span>
                    </div>
                    <div class="metric">
                        <strong>F1 Score</strong>
                        <span>{{ "%.4f"|format(iteration.get('f1', 0)) if iteration.get('f1') is not none else 'N/A' }}</span>
                    </div>
                </div>
                
                <h4>Prompt Used</h4>
                <pre class="prompt">{{ iteration.get('prompt', 'N/A') }}</pre>
                
                <div class="analysis-section">
                    {% if iteration.get('correct_analysis') %}
                    <h3>Correct Predictions Analysis</h3>
                    <pre class="analysis">{{ iteration.get('correct_analysis') }}</pre>
                    {% endif %}
                    
                    {% if iteration.get('incorrect_analysis') %}
                    <h3>Incorrect Predictions Analysis</h3>
                    <pre class="analysis">{{ iteration.get('incorrect_analysis') }}</pre>
                    {% endif %}

                    {% if iteration.get('fp_analysis') %}
                    <h3>False Positives Analysis</h3>
                    <pre class="analysis">{{ iteration.get('fp_analysis') }}</pre>
                    {% endif %}

                    {% if iteration.get('fn_analysis') %}
                    <h3>False Negatives Analysis</h3>
                    <pre class="analysis">{{ iteration.get('fn_analysis') }}</pre>
                    {% endif %}

                    {% if iteration.get('tp_analysis') %}
                    <h3>True Positives Analysis</h3>
                    <pre class="analysis">{{ iteration.get('tp_analysis') }}</pre>
                    {% endif %}

                    {% if iteration.get('invalid_analysis') %}
                    <h3>Invalid Outputs Analysis</h3>
                    <pre class="analysis">{{ iteration.get('invalid_analysis') }}</pre>
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