import unittest
from unittest.mock import patch
from src.iterative_prompt_optimization.utils import estimate_cost

class TestEstimateCost(unittest.TestCase):
    @patch('src.iterative_prompt_optimization.utils.PRICING', {'test_provider': {'test_model': 0.01}})
    def test_estimate_cost(self):
        token_usage = 1000
        provider = 'test_provider'
        model = 'test_model'
        estimated_cost = estimate_cost(token_usage, provider, model)
        self.assertEqual(estimated_cost, '$0.01')

        # Test with different token usage
        token_usage = 5000
        estimated_cost = estimate_cost(token_usage, provider, model)
        self.assertEqual(estimated_cost, '$0.05')

        # Test with zero tokens
        token_usage = 0
        estimated_cost = estimate_cost(token_usage, provider, model)
        self.assertEqual(estimated_cost, '$0.00')

    @patch('src.iterative_prompt_optimization.utils.PRICING', {'another_provider': {'another_model': 0.02}})
    def test_estimate_cost_different_provider(self):
        token_usage = 1000
        provider = 'another_provider'
        model = 'another_model'
        estimated_cost = estimate_cost(token_usage, provider, model)
        self.assertEqual(estimated_cost, '$0.02')

if __name__ == '__main__':
    unittest.main()