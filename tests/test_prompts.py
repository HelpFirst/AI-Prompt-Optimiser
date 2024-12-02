import pytest
from src import prompts

def test_prompt_engineer_input_template():
    """Test the prompt engineer input template formatting"""
    test_data = {
        'initial_prompt': 'Test prompt',
        'precision': 0.8,
        'recall': 0.75,
        'accuracy': 0.85,
        'f1': 0.77,
        'total_predictions': 100,
        'valid_predictions': 95,
        'invalid_predictions': 5,
        'tp_analysis': 'Good patterns',
        'fp_analysis': 'False positive issues',
        'fn_analysis': 'False negative issues',
        'invalid_analysis': 'Invalid output issues',
        'output_format_prompt': 'Format instructions',
        'prompt_engineering_comments': 'Additional comments'
    }
    
    formatted_prompt = prompts.PROMPT_ENGINEER_INPUT.format(**test_data)
    
    # Check that all placeholders were replaced
    assert '{initial_prompt}' not in formatted_prompt
    assert '{precision}' not in formatted_prompt
    assert '{recall}' not in formatted_prompt
    assert '{accuracy}' not in formatted_prompt
    assert '{f1}' not in formatted_prompt
    assert '{total_predictions}' not in formatted_prompt
    assert '{valid_predictions}' not in formatted_prompt
    assert '{invalid_predictions}' not in formatted_prompt
    assert '{tp_analysis}' not in formatted_prompt
    assert '{fp_analysis}' not in formatted_prompt
    assert '{fn_analysis}' not in formatted_prompt
    assert '{invalid_analysis}' not in formatted_prompt
    assert '{output_format_prompt}' not in formatted_prompt
    assert '{prompt_engineering_comments}' not in formatted_prompt
    
    # Check that test data values are present
    assert 'Test prompt' in formatted_prompt
    assert '0.8' in formatted_prompt
    assert '0.75' in formatted_prompt
    assert '0.85' in formatted_prompt
    assert '0.77' in formatted_prompt
    assert 'Good patterns' in formatted_prompt
    assert 'False positive issues' in formatted_prompt
    assert 'False negative issues' in formatted_prompt
    assert 'Invalid output issues' in formatted_prompt
    assert 'Format instructions' in formatted_prompt
    assert 'Additional comments' in formatted_prompt

def test_prompt_engineer_input_missing_data():
    """Test the prompt engineer input template with missing data"""
    test_data = {
        'initial_prompt': 'Test prompt',
        'precision': 0.8,
        'recall': 0.75,
        # Missing accuracy
        'f1': 0.77,
        'total_predictions': 100,
        'valid_predictions': 95,
        'invalid_predictions': 5,
        'tp_analysis': 'Good patterns',
        'fp_analysis': 'False positive issues',
        'fn_analysis': 'False negative issues',
        'invalid_analysis': 'Invalid output issues',
        'output_format_prompt': 'Format instructions',
        'prompt_engineering_comments': 'Additional comments'
    }
    
    with pytest.raises(KeyError):
        prompts.PROMPT_ENGINEER_INPUT.format(**test_data)

def test_prompt_engineer_input_empty_values():
    """Test the prompt engineer input template with empty values"""
    test_data = {
        'initial_prompt': '',
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'f1': 0.0,
        'total_predictions': 0,
        'valid_predictions': 0,
        'invalid_predictions': 0,
        'tp_analysis': '',
        'fp_analysis': '',
        'fn_analysis': '',
        'invalid_analysis': '',
        'output_format_prompt': '',
        'prompt_engineering_comments': ''
    }
    
    formatted_prompt = prompts.PROMPT_ENGINEER_INPUT.format(**test_data)
    
    # Check that the template still formats correctly with empty values
    assert formatted_prompt is not None
    assert len(formatted_prompt) > 0
    assert '0.0' in formatted_prompt  # Check numeric zero values
    assert '0' in formatted_prompt    # Check integer zero values

def test_prompt_constants_exist():
    """Test that all required prompt constants exist"""
    required_prompts = [
        'PROMPT_ENGINEER_INPUT',
        'ANALYZE_FALSE_POSITIVES',
        'ANALYZE_FALSE_NEGATIVES',
        'ANALYZE_TRUE_POSITIVES',
        'ANALYZE_INVALID_OUTPUTS',
        'VALIDATE_AND_IMPROVE_PROMPT'
    ]
    
    for prompt_name in required_prompts:
        assert hasattr(prompts, prompt_name), f"Missing prompt constant: {prompt_name}"
        prompt_value = getattr(prompts, prompt_name)
        assert isinstance(prompt_value, str), f"Prompt {prompt_name} should be a string"
        assert len(prompt_value) > 0, f"Prompt {prompt_name} should not be empty"

def test_prompt_placeholders_consistency():
    """Test that all prompt templates have consistent placeholder names"""
    # Common placeholders that should be used consistently across prompts
    common_placeholders = [
        '{initial_prompt}',
        '{output_format_prompt}',
        '{examples}',
        '{analysis}'
    ]
    
    # Check ANALYZE prompts for consistency
    analysis_prompts = [
        prompts.ANALYZE_FALSE_POSITIVES,
        prompts.ANALYZE_FALSE_NEGATIVES,
        prompts.ANALYZE_TRUE_POSITIVES,
        prompts.ANALYZE_INVALID_OUTPUTS
    ]
    
    for prompt in analysis_prompts:
        assert '{examples}' in prompt, "Analysis prompt missing {examples} placeholder"
        assert '{output_format_prompt}' in prompt, "Analysis prompt missing {output_format_prompt} placeholder"
    
    # Check VALIDATE_AND_IMPROVE_PROMPT
    assert '{initial_prompt}' in prompts.VALIDATE_AND_IMPROVE_PROMPT, "Validation prompt missing {initial_prompt} placeholder"
    assert '{output_format_prompt}' in prompts.VALIDATE_AND_IMPROVE_PROMPT, "Validation prompt missing {output_format_prompt} placeholder"

def test_prompt_formatting_with_special_characters():
    """Test prompt formatting with special characters in the input"""
    test_data = {
        'initial_prompt': 'Test\nprompt\nwith\nspecial\nchars\n\t!@#$%^&*()',
        'precision': 0.8,
        'recall': 0.75,
        'accuracy': 0.85,
        'f1': 0.77,
        'total_predictions': 100,
        'valid_predictions': 95,
        'invalid_predictions': 5,
        'tp_analysis': 'Good\npatterns\n!@#',
        'fp_analysis': 'False\npositive\nissues\n$%^',
        'fn_analysis': 'False\nnegative\nissues\n&*()',
        'invalid_analysis': 'Invalid\noutput\nissues\n\\n\\t',
        'output_format_prompt': 'Format\ninstructions\n`~',
        'prompt_engineering_comments': 'Additional\ncomments\n{}[]'
    }
    
    formatted_prompt = prompts.PROMPT_ENGINEER_INPUT.format(**test_data)
    
    # Check that special characters were preserved
    assert '!@#$%^&*()' in formatted_prompt
    assert '\\n\\t' in formatted_prompt
    assert '`~' in formatted_prompt
    assert '{}[]' in formatted_prompt 