import pytest
import os
import json
import pandas as pd
import numpy as np
from src import dashboard_generator

@pytest.fixture
def sample_metrics():
    """Sample metrics data for testing"""
    return {
        'precision': 0.85,
        'recall': 0.78,
        'accuracy': 0.82,
        'f1': 0.81,
        'total_examples': 100,
        'processed_examples': 95,
        'failed_count': 5,
        'predictions': [1, 0, 1, 1, 0],
        'chain_of_thought': ['Reasoning 1', 'Reasoning 2', 'Reasoning 3', 'Reasoning 4', 'Reasoning 5'],
        'raw_outputs': ['Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5']
    }

@pytest.fixture
def sample_iteration_data():
    """Sample iteration data for testing"""
    return {
        'iteration': 1,
        'prompt': 'Test prompt',
        'metrics': {
            'precision': 0.85,
            'recall': 0.78,
            'accuracy': 0.82,
            'f1': 0.81
        },
        'examples': [
            {'text': 'Example 1', 'prediction': 1, 'actual': 1, 'chain_of_thought': 'Reasoning 1'},
            {'text': 'Example 2', 'prediction': 0, 'actual': 0, 'chain_of_thought': 'Reasoning 2'}
        ]
    }

def test_generate_iteration_dashboard(sample_metrics, tmp_path):
    """Test generation of iteration dashboard"""
    iteration = 1
    prompt = "Test prompt for iteration 1"
    output_dir = str(tmp_path)
    
    dashboard = dashboard_generator.generate_iteration_dashboard(
        iteration=iteration,
        prompt=prompt,
        metrics=sample_metrics,
        output_dir=output_dir
    )
    
    assert isinstance(dashboard, str)
    assert dashboard.endswith('.html')
    assert os.path.exists(os.path.join(output_dir, dashboard))
    
    # Check content of generated file
    with open(os.path.join(output_dir, dashboard), 'r') as f:
        content = f.read()
        assert 'Test prompt for iteration 1' in content
        assert 'Precision: 0.85' in content
        assert 'Recall: 0.78' in content
        assert 'Accuracy: 0.82' in content
        assert 'F1 Score: 0.81' in content

def test_generate_combined_dashboard(sample_iteration_data, tmp_path):
    """Test generation of combined dashboard"""
    iterations_data = [sample_iteration_data]
    output_dir = str(tmp_path)
    
    dashboard = dashboard_generator.generate_combined_dashboard(
        iterations_data=iterations_data,
        output_dir=output_dir
    )
    
    assert isinstance(dashboard, str)
    assert dashboard.endswith('.html')
    assert os.path.exists(os.path.join(output_dir, dashboard))
    
    # Check content of generated file
    with open(os.path.join(output_dir, dashboard), 'r') as f:
        content = f.read()
        assert 'Combined Performance Dashboard' in content
        assert 'Metrics Over Time' in content
        assert 'Test prompt' in content

def test_metrics_visualization(sample_metrics):
    """Test metrics visualization generation"""
    fig = dashboard_generator.create_metrics_visualization([sample_metrics])
    
    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Should have at least one trace

def test_confusion_matrix_visualization(sample_metrics):
    """Test confusion matrix visualization generation"""
    y_true = [1, 0, 1, 1, 0]
    y_pred = sample_metrics['predictions']
    
    fig = dashboard_generator.create_confusion_matrix_plot(y_true, y_pred)
    
    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_dashboard_with_empty_data(tmp_path):
    """Test dashboard generation with empty data"""
    empty_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'f1': 0.0,
        'total_examples': 0,
        'processed_examples': 0,
        'failed_count': 0,
        'predictions': [],
        'chain_of_thought': [],
        'raw_outputs': []
    }
    
    output_dir = str(tmp_path)
    
    # Should handle empty data without errors
    dashboard = dashboard_generator.generate_iteration_dashboard(
        iteration=1,
        prompt="Empty test",
        metrics=empty_metrics,
        output_dir=output_dir
    )
    
    assert isinstance(dashboard, str)
    assert os.path.exists(os.path.join(output_dir, dashboard))

def test_dashboard_with_special_characters(sample_metrics, tmp_path):
    """Test dashboard generation with special characters in prompt"""
    special_prompt = "Test prompt with <special> & 'characters' & \"quotes\""
    output_dir = str(tmp_path)
    
    dashboard = dashboard_generator.generate_iteration_dashboard(
        iteration=1,
        prompt=special_prompt,
        metrics=sample_metrics,
        output_dir=output_dir
    )
    
    assert isinstance(dashboard, str)
    assert os.path.exists(os.path.join(output_dir, dashboard))
    
    # Check that special characters are properly escaped
    with open(os.path.join(output_dir, dashboard), 'r') as f:
        content = f.read()
        assert '&lt;special&gt;' in content
        assert '&amp;' in content
        assert 'quotes' in content

def test_metrics_over_time_visualization():
    """Test metrics over time visualization"""
    iterations_data = [
        {
            'iteration': i,
            'metrics': {
                'precision': 0.8 + i/10,
                'recall': 0.7 + i/10,
                'accuracy': 0.75 + i/10,
                'f1': 0.77 + i/10
            }
        } for i in range(3)
    ]
    
    fig = dashboard_generator.create_metrics_over_time_plot(iterations_data)
    
    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Should have 4 metrics traces

def test_example_visualization(sample_metrics):
    """Test example visualization generation"""
    examples = [
        {'text': 'Example 1', 'prediction': 1, 'actual': 1},
        {'text': 'Example 2', 'prediction': 0, 'actual': 0}
    ]
    
    visualization = dashboard_generator.create_examples_visualization(examples)
    
    assert visualization is not None
    assert isinstance(visualization, str)
    assert 'Example 1' in visualization
    assert 'Example 2' in visualization

def test_dashboard_file_naming(sample_metrics, tmp_path):
    """Test dashboard file naming convention"""
    output_dir = str(tmp_path)
    
    # Generate multiple dashboards
    dashboards = []
    for i in range(3):
        dashboard = dashboard_generator.generate_iteration_dashboard(
            iteration=i,
            prompt=f"Test prompt {i}",
            metrics=sample_metrics,
            output_dir=output_dir
        )
        dashboards.append(dashboard)
    
    # Check unique naming
    assert len(set(dashboards)) == 3  # All filenames should be unique
    for dashboard in dashboards:
        assert dashboard.startswith('iteration_')
        assert dashboard.endswith('.html') 