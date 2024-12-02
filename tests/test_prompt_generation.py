import pytest
from unittest.mock import patch, MagicMock
from src import prompt_generation
import os
from pathlib import Path

@pytest.fixture
def sample_metrics():
    return {
        'precision': 0.8,
        'recall': 0.75,
        'accuracy': 0.85,
        'f1': 0.77,
        'total_predictions': 100,
        'valid_predictions': 95,
        'invalid_predictions': 5
    }

@pytest.fixture
def sample_data():
    return {
        'false_positives': [
            {'text': 'False positive example', 'chain_of_thought': 'Reasoning for FP'}
        ],
        'false_negatives': [
            {'text': 'False negative example', 'chain_of_thought': 'Reasoning for FN'}
        ],
        'true_positives': [
            {'text': 'True positive example', 'chain_of_thought': 'Reasoning for TP'}
        ],
        'invalid_outputs': [
            {'text': 'Invalid output example', 'raw_output': 'Invalid format'}
        ]
    }

@patch('src.prompt_generation.get_analysis')
def test_generate_new_prompt(mock_get_analysis, sample_metrics, sample_data, tmp_path):
    """Test the generation of a new prompt"""
    # Mock the analysis responses
    mock_get_analysis.side_effect = [
        "Analysis of false positives",  # FP analysis
        "Analysis of false negatives",  # FN analysis
        "Analysis of true positives",   # TP analysis
        "Analysis of invalid outputs",  # Invalid analysis
        "New improved prompt"           # Final prompt
    ]
    
    initial_prompt = "Initial test prompt"
    output_format_prompt = "Output format instructions"
    
    new_prompt, analyses, prompts_used = prompt_generation.generate_new_prompt(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        false_positives=sample_data['false_positives'],
        false_negatives=sample_data['false_negatives'],
        true_positives=sample_data['true_positives'],
        invalid_outputs=sample_data['invalid_outputs'],
        previous_metrics=sample_metrics,
        log_dir=str(tmp_path),
        iteration=1,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    # Verify the results
    assert isinstance(new_prompt, str)
    assert new_prompt == "New improved prompt"
    assert isinstance(analyses, dict)
    assert isinstance(prompts_used, dict)
    assert mock_get_analysis.call_count == 5

@patch('src.prompt_generation.get_analysis')
def test_generate_new_prompt_no_issues(mock_get_analysis, sample_metrics):
    """Test prompt generation when there are no issues to analyze"""
    mock_get_analysis.return_value = "New prompt with no issues"
    
    new_prompt, analyses, prompts_used = prompt_generation.generate_new_prompt(
        initial_prompt="Test prompt",
        output_format_prompt="Test format",
        false_positives=[],
        false_negatives=[],
        true_positives=[],
        invalid_outputs=[],
        previous_metrics=sample_metrics,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    assert analyses['fp_analysis'] == "No false positives found in this iteration."
    assert analyses['fn_analysis'] == "No false negatives found in this iteration."
    assert analyses['tp_analysis'] == "No true positives found in this iteration."
    assert analyses['invalid_analysis'] == "No invalid outputs found in this iteration."
    assert new_prompt == "New prompt with no issues"

@patch('src.prompt_generation.get_analysis')
def test_validate_and_improve_prompt_valid(mock_get_analysis):
    """Test prompt validation when the prompt is valid"""
    mock_get_analysis.return_value = "Validation: VALID\nNo improvements needed."
    
    original_prompt = "Test prompt"
    output_format = "Test format"
    
    improved_prompt, validation_result = prompt_generation.validate_and_improve_prompt(
        new_prompt=original_prompt,
        output_format_prompt=output_format,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    assert improved_prompt == original_prompt
    assert validation_result == "Validation: VALID\nNo improvements needed."
    mock_get_analysis.assert_called_once()

@patch('src.prompt_generation.get_analysis')
def test_validate_and_improve_prompt_needs_improvement(mock_get_analysis):
    """Test prompt validation when the prompt needs improvement"""
    mock_get_analysis.return_value = (
        "Validation: NEEDS IMPROVEMENT\n"
        "Improved Prompt: Better test prompt\n"
        "Explanation: Made improvements"
    )
    
    original_prompt = "Test prompt"
    output_format = "Test format"
    
    improved_prompt, validation_result = prompt_generation.validate_and_improve_prompt(
        new_prompt=original_prompt,
        output_format_prompt=output_format,
        provider="test_provider",
        model="test_model",
        temperature=0.7
    )
    
    assert improved_prompt == "Better test prompt"
    assert "NEEDS IMPROVEMENT" in validation_result
    mock_get_analysis.assert_called_once()

def test_prompt_generation_with_comments(sample_data, sample_metrics):
    """Test prompt generation with additional comments"""
    with patch('src.prompt_generation.get_analysis') as mock_get_analysis:
        mock_get_analysis.side_effect = [
            "FP Analysis",      # False positives analysis
            "FN Analysis",      # False negatives analysis
            "TP Analysis",      # True positives analysis
            "Invalid Analysis", # Invalid outputs analysis
            "New prompt with comments"  # Final prompt
        ]
        
        new_prompt, analyses, prompts_used = prompt_generation.generate_new_prompt(
            initial_prompt="Test prompt",
            output_format_prompt="Test format",
            false_positives=sample_data['false_positives'],
            false_negatives=sample_data['false_negatives'],
            true_positives=sample_data['true_positives'],
            invalid_outputs=sample_data['invalid_outputs'],
            previous_metrics=sample_metrics,
            fp_comments="FP comment",
            fn_comments="FN comment",
            tp_comments="TP comment",
            invalid_comments="Invalid comment",
            prompt_engineering_comments="Engineering comment",
            provider="test_provider",
            model="test_model",
            temperature=0.7
        )
        
        assert new_prompt == "New prompt with comments"
        # Verify that analyses were generated
        assert analyses['fp_analysis'] == "FP Analysis"
        assert analyses['fn_analysis'] == "FN Analysis"
        assert analyses['tp_analysis'] == "TP Analysis"
        assert analyses['invalid_analysis'] == "Invalid Analysis"
        # Verify that comments were included in the prompt engineering input
        assert "Engineering comment" in prompts_used['prompt_engineer_input']