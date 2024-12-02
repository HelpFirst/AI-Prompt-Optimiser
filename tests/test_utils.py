import pytest
from src import utils
import pandas as pd

@pytest.fixture
def sample_eval_data():
    return pd.DataFrame({
        'text': ['This is a test', 'Another test example', 'Third test case']
    })

def test_estimate_token_usage():
    """Test the token estimation functionality"""
    initial_prompt = "Classify the following text:"
    output_format_prompt = "Respond with yes or no"
    eval_data = pd.DataFrame({
        'text': ['Sample text for testing']
    })
    iterations = 3
    
    # Test that the function runs without errors and returns an integer
    token_estimate = utils.estimate_token_usage(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        eval_data=eval_data,
        iterations=iterations
    )
    assert isinstance(token_estimate, int)
    assert token_estimate > 0

def test_estimate_token_usage_empty_input():
    """Test token estimation with empty inputs"""
    initial_prompt = ""
    output_format_prompt = ""
    eval_data = pd.DataFrame({'text': []})
    iterations = 1
    
    token_estimate = utils.estimate_token_usage(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        eval_data=eval_data,
        iterations=iterations
    )
    assert isinstance(token_estimate, int)
    assert token_estimate >= 0

def test_estimate_token_usage_non_string_input():
    """Test token estimation with non-string inputs in eval data"""
    initial_prompt = "Test prompt"
    output_format_prompt = "Test format"
    eval_data = pd.DataFrame({
        'text': [123, "valid string", None]  # Mixed types
    })
    iterations = 1
    
    # Should handle non-string inputs gracefully
    token_estimate = utils.estimate_token_usage(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        eval_data=eval_data,
        iterations=iterations
    )
    assert isinstance(token_estimate, int) 