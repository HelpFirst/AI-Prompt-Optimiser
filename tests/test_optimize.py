import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from src import optimize

@pytest.fixture
def sample_eval_data():
    """Sample evaluation data for testing"""
    return pd.DataFrame({
        'text': ['Positive example', 'Negative example', 'Another positive'],
        'label': [1, 0, 1]
    })

@pytest.fixture
def sample_output_schema():
    """Sample output schema for binary classification"""
    return {
        'prediction': {'type': 'integer', 'enum': [0, 1]},
        'chain_of_thought': {'type': 'string'}
    }

@pytest.fixture
def mock_evaluation_results():
    """Mock evaluation results"""
    return {
        'precision': 0.8,
        'recall': 0.75,
        'accuracy': 0.85,
        'f1': 0.77,
        'total_examples': 100,
        'processed_examples': 95,
        'failed_count': 5,
        'predictions': [1, 0, 1],
        'chain_of_thought': ['Reasoning 1', 'Reasoning 2', 'Reasoning 3'],
        'raw_outputs': ['Output 1', 'Output 2', 'Output 3']
    }

@patch('src.optimize.generate_new_prompt')
@patch('src.optimize.evaluate_prompt')
def test_optimize_prompt_binary(mock_evaluate, mock_generate, sample_eval_data, sample_output_schema, mock_evaluation_results):
    """Test prompt optimization for binary classification"""
    mock_evaluate.return_value = mock_evaluation_results
    mock_generate.return_value = ("New prompt", {}, {})
    
    result = optimize.optimize_prompt(
        initial_prompt="Test prompt",
        output_format_prompt="Test format",
        eval_data=sample_eval_data,
        iterations=2,
        eval_provider="test_provider",
        eval_model="test_model",
        eval_temperature=0.7,
        output_schema=sample_output_schema,
        use_cache=True
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 4  # best_prompt, best_metrics, all_metrics, dashboards
    assert isinstance(result[0], str)  # best_prompt
    assert isinstance(result[1], dict)  # best_metrics
    assert isinstance(result[2], list)  # all_metrics
    assert isinstance(result[3], dict)  # dashboards

@patch('src.optimize.generate_new_prompt_multiclass')
@patch('src.optimize.evaluate_prompt')
def test_optimize_prompt_multiclass(mock_evaluate, mock_generate, sample_output_schema):
    """Test prompt optimization for multiclass classification"""
    # Create multiclass test data
    eval_data = pd.DataFrame({
        'text': ['Class A', 'Class B', 'Class C'],
        'label': [0, 1, 2]
    })
    
    mock_evaluate.return_value = {
        **mock_evaluation_results,
        'confusion_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
    mock_generate.return_value = ("New multiclass prompt", {}, {})
    
    result = optimize.optimize_prompt(
        initial_prompt="Test multiclass prompt",
        output_format_prompt="Test format",
        eval_data=eval_data,
        iterations=2,
        eval_provider="test_provider",
        eval_model="test_model",
        eval_temperature=0.7,
        output_schema={**sample_output_schema, 'prediction': {'type': 'integer', 'enum': [0, 1, 2]}},
        use_cache=True
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert 'confusion_matrix' in result[1]  # Check for multiclass-specific metrics

@patch('src.optimize.validate_and_improve_prompt')
@patch('src.optimize.evaluate_prompt')
def test_optimize_prompt_with_validation(mock_evaluate, mock_validate, sample_eval_data, sample_output_schema, mock_evaluation_results):
    """Test prompt optimization with validation step"""
    mock_evaluate.return_value = mock_evaluation_results
    mock_validate.return_value = ("Improved prompt", "Validation result")
    
    result = optimize.optimize_prompt(
        initial_prompt="Test prompt",
        output_format_prompt="Test format",
        eval_data=sample_eval_data,
        iterations=1,
        eval_provider="test_provider",
        eval_model="test_model",
        output_schema=sample_output_schema,
        skip_prompt_validation=False
    )
    
    assert mock_validate.called
    assert isinstance(result[0], str)
    assert "Improved prompt" in result[0]

def test_optimize_prompt_with_comments(sample_eval_data, sample_output_schema):
    """Test prompt optimization with additional comments"""
    with patch('src.optimize.evaluate_prompt') as mock_evaluate, \
         patch('src.optimize.generate_new_prompt') as mock_generate:
        
        mock_evaluate.return_value = mock_evaluation_results
        mock_generate.return_value = ("New prompt with comments", {}, {})
        
        result = optimize.optimize_prompt(
            initial_prompt="Test prompt",
            output_format_prompt="Test format",
            eval_data=sample_eval_data,
            iterations=1,
            eval_provider="test_provider",
            eval_model="test_model",
            output_schema=sample_output_schema,
            fp_comments="FP comment",
            fn_comments="FN comment",
            tp_comments="TP comment",
            invalid_comments="Invalid comment",
            prompt_engineering_comments="Engineering comment"
        )
        
        # Verify comments were passed to generate_new_prompt
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs['fp_comments'] == "FP comment"
        assert call_kwargs['fn_comments'] == "FN comment"
        assert call_kwargs['tp_comments'] == "TP comment"
        assert call_kwargs['invalid_comments'] == "Invalid comment"
        assert call_kwargs['prompt_engineering_comments'] == "Engineering comment"

@patch('src.optimize.evaluate_prompt')
def test_optimize_prompt_error_handling(mock_evaluate, sample_eval_data, sample_output_schema):
    """Test error handling in prompt optimization"""
    mock_evaluate.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        optimize.optimize_prompt(
            initial_prompt="Test prompt",
            output_format_prompt="Test format",
            eval_data=sample_eval_data,
            iterations=1,
            eval_provider="test_provider",
            eval_model="test_model",
            output_schema=sample_output_schema
        )
    
    assert "Test error" in str(exc_info.value)

@patch('src.optimize.generate_iteration_dashboard')
@patch('src.optimize.generate_combined_dashboard')
@patch('src.optimize.evaluate_prompt')
def test_optimize_prompt_dashboard_generation(mock_evaluate, mock_combined_dash, mock_iter_dash, 
                                           sample_eval_data, sample_output_schema, mock_evaluation_results):
    """Test dashboard generation during optimization"""
    mock_evaluate.return_value = mock_evaluation_results
    mock_iter_dash.return_value = "Iteration Dashboard"
    mock_combined_dash.return_value = "Combined Dashboard"
    
    result = optimize.optimize_prompt(
        initial_prompt="Test prompt",
        output_format_prompt="Test format",
        eval_data=sample_eval_data,
        iterations=2,
        eval_provider="test_provider",
        eval_model="test_model",
        output_schema=sample_output_schema,
        experiment_name="test_experiment"
    )
    
    assert mock_iter_dash.call_count == 2  # Called for each iteration
    assert mock_combined_dash.called
    assert 'iteration_dashboards' in result[3]
    assert 'combined_dashboard' in result[3] 