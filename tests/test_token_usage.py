import unittest
import pandas as pd
from src.iterative_prompt_optimization.utils import estimate_token_usage

class TestEstimateTokenUsage(unittest.TestCase):
    def test_estimate_token_usage(self):
        prompt = "This is a test prompt"
        eval_data = pd.DataFrame({'text': ['This is a test', 'Another test']})
        iterations = 3
        output_format_prompt = "Test output format"

        estimated_tokens = estimate_token_usage(prompt, output_format_prompt, eval_data, iterations)

        # Assert that the estimated token usage is greater than zero
        self.assertGreater(estimated_tokens, 0)

        # Test with empty data
        empty_data = pd.DataFrame({'text': []})
        estimated_tokens_empty = estimate_token_usage(prompt, output_format_prompt, empty_data, iterations)
        self.assertGreater(estimated_tokens_empty, 0)

        # Test with longer prompt and more iterations
        long_prompt = "This is a much longer prompt with more words to test the estimation"
        estimated_tokens_long = estimate_token_usage(long_prompt, output_format_prompt, eval_data, 10)
        self.assertGreater(estimated_tokens_long, estimated_tokens)

if __name__ == '__main__':
    unittest.main()