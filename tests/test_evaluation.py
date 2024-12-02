import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import json
from src import evaluation
from src.utils import transform_and_compare_output

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
def mock_model_outputs():
    """Sample model outputs for testing"""
    return [
        {'choices': [{'message': {'content': json.dumps({'prediction': 1, 'chain_of_thought': 'Reasoning 1'})}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 0, 'chain_of_thought': 'Reasoning 2'})}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 1, 'chain_of_thought': 'Reasoning 3'})}}]}
    ]

@patch('src.evaluation.transform_and_compare_output')
@patch('src.evaluation.get_model_output')
def test_evaluate_prompt_binary(mock_get_output, mock_transform, sample_eval_data, sample_output_schema, mock_model_outputs, tmp_path):
    """Test prompt evaluation for binary classification"""
    mock_get_output.side_effect = mock_model_outputs
    # Return (prediction, is_correct, is_valid, chain_of_thought)
    mock_transform.side_effect = [
        (1, True, True, "Reasoning 1"),
        (0, True, True, "Reasoning 2"),
        (1, True, True, "Reasoning 3")
    ]
    
    results = evaluation.evaluate_prompt(
        full_prompt="Test prompt",
        eval_data=sample_eval_data,
        output_schema=sample_output_schema,
        problem_type="binary",
        log_dir=str(tmp_path),
        iteration=1,
        use_cache=True,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    # Check that all expected metrics are present
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'precision', 'recall', 'accuracy', 'f1',
        'total_examples', 'processed_examples', 'failed_count',
        'predictions', 'chain_of_thought', 'raw_outputs'
    ])
    
    # Check metric values
    assert results['total_examples'] == 3
    assert results['processed_examples'] == 3
    assert results['failed_count'] == 0
    assert 0 <= results['precision'] <= 1
    assert 0 <= results['recall'] <= 1
    assert 0 <= results['accuracy'] <= 1
    assert 0 <= results['f1'] <= 1

@patch('src.evaluation.transform_and_compare_output')
@patch('src.evaluation.get_model_output')
def test_evaluate_prompt_with_invalid_outputs(mock_get_output, mock_transform, sample_eval_data, sample_output_schema):
    """Test handling of invalid model outputs"""
    # Mock outputs with some invalid ones
    invalid_outputs = [
        {'choices': [{'message': {'content': 'Invalid JSON'}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 'invalid', 'chain_of_thought': 'Reasoning'})}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 1, 'chain_of_thought': 'Valid'})}}]}
    ]
    mock_get_output.side_effect = invalid_outputs
    mock_transform.side_effect = [
        (None, False, False, None),
        (None, False, False, None),
        (1, True, True, "Valid")
    ]
    
    results = evaluation.evaluate_prompt(
        full_prompt="Test prompt",
        eval_data=sample_eval_data,
        output_schema=sample_output_schema,
        problem_type="binary",
        use_cache=False,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    assert results['total_examples'] == 3
    assert results['processed_examples'] == 3
    assert results['failed_count'] == 0
    assert len(results['predictions']) == 1  # Only one valid prediction

@patch('src.evaluation.transform_and_compare_output')
@patch('src.evaluation.get_model_output')
def test_evaluate_prompt_multiclass(mock_get_output, mock_transform):
    """Test prompt evaluation for multiclass classification"""
    # Create multiclass test data
    eval_data = pd.DataFrame({
        'text': ['Class A text', 'Class B text', 'Class C text'],
        'label': [0, 1, 2]
    })
    
    output_schema = {
        'prediction': {'type': 'integer', 'enum': [0, 1, 2]},
        'chain_of_thought': {'type': 'string'}
    }
    
    outputs = [
        {'choices': [{'message': {'content': json.dumps({'prediction': 0, 'chain_of_thought': 'Reasoning A'})}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 1, 'chain_of_thought': 'Reasoning B'})}}]},
        {'choices': [{'message': {'content': json.dumps({'prediction': 2, 'chain_of_thought': 'Reasoning C'})}}]}
    ]
    mock_get_output.side_effect = outputs
    mock_transform.side_effect = [
        (0, True, True, "Reasoning A"),
        (1, True, True, "Reasoning B"),
        (2, True, True, "Reasoning C")
    ]
    
    results = evaluation.evaluate_prompt(
        full_prompt="Test prompt",
        eval_data=eval_data,
        output_schema=output_schema,
        problem_type="multiclass",
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    assert results['total_examples'] == 3
    assert results['processed_examples'] == 3
    assert results['failed_count'] == 0
    assert len(results['predictions']) == 3
    assert 'confusion_matrix' in results

@patch('src.evaluation.transform_and_compare_output')
@patch('src.evaluation.get_model_output')
def test_evaluate_prompt_with_logging(mock_get_output, mock_transform, sample_eval_data, sample_output_schema, tmp_path):
    """Test that results are properly logged"""
    outputs = [
        {'choices': [{'message': {'content': json.dumps({'prediction': 1, 'chain_of_thought': 'Reasoning'})}}]}
        for _ in range(len(sample_eval_data))
    ]
    mock_get_output.side_effect = outputs
    mock_transform.side_effect = [
        (1, True, True, "Reasoning")
        for _ in range(len(sample_eval_data))
    ]
    
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    results = evaluation.evaluate_prompt(
        full_prompt="Test prompt",
        eval_data=sample_eval_data,
        output_schema=sample_output_schema,
        problem_type="binary",
        log_dir=str(log_dir),
        iteration=1,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    # Check that log files were created
    assert any(log_dir.iterdir())
    
    # Check log file content
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) > 0
    
    with open(log_files[0]) as f:
        log_data = json.load(f)
        assert isinstance(log_data, dict)
        assert 'evaluations' in log_data
        assert len(log_data['evaluations']) == len(sample_eval_data)

def test_evaluate_prompt_with_failed_examples(sample_eval_data, sample_output_schema):
    """Test handling of failed examples during evaluation"""
    with patch('src.evaluation.get_model_output') as mock_get_output:
        # Simulate an error in processing
        mock_get_output.side_effect = Exception("Test error")
        
        results = evaluation.evaluate_prompt(
            full_prompt="Test prompt",
            eval_data=sample_eval_data,
            output_schema=sample_output_schema,
            problem_type="binary",
            provider="test_provider",
            model="test_model",
            temperature=0.7
        )
        
        assert results['total_examples'] == 3
        assert results['failed_count'] == 3
        assert len(results['failed_examples']) == 3
        assert all('error' in ex for ex in results['failed_examples']) 