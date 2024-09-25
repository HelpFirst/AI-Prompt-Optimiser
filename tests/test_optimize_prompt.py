import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.iterative_prompt_optimization.optimize import optimize_prompt
from src.iterative_prompt_optimization.evaluation import evaluate_prompt
from src.iterative_prompt_optimization.prompt_generation import generate_new_prompt
from src.iterative_prompt_optimization.utils import (
    estimate_token_usage, estimate_cost, display_best_prompt,
    display_comparison_table, log_final_results, select_model,
    create_log_directory, log_initial_setup, display_metrics,
    create_metric_entry, update_best_metrics
)

class TestOptimizePrompt(unittest.TestCase):

    def setUp(self):
        self.initial_prompt = "Test initial prompt"
        self.output_format_prompt = "Test output format prompt"
        self.eval_data = pd.DataFrame({
            'text': ['positive text', 'negative text'],
            'label': [1, 0]
        })
        self.iterations = 2

    @patch('src.iterative_prompt_optimization.optimize.evaluate_prompt')
    @patch('src.iterative_prompt_optimization.optimize.generate_new_prompt')
    @patch('src.iterative_prompt_optimization.optimize.display_metrics')
    @patch('src.iterative_prompt_optimization.optimize.display_best_prompt')
    @patch('src.iterative_prompt_optimization.optimize.display_comparison_table')
    @patch('src.iterative_prompt_optimization.optimize.log_final_results')
    def test_optimize_prompt(self, mock_log_final_results, mock_display_comparison_table, 
                             mock_display_best_prompt, mock_display_metrics, 
                             mock_generate_new_prompt, mock_evaluate_prompt):
        # Mock the evaluate_prompt function to return some results
        mock_evaluate_prompt.return_value = {
            'precision': 0.8,
            'recall': 0.7,
            'accuracy': 0.75,
            'f1': 0.74,
            'invalid_predictions': 0,
            'false_positives': [],
            'false_negatives': []
        }

        # Mock the generate_new_prompt function to return a new prompt
        mock_generate_new_prompt.return_value = "New test prompt"

        # Call the function we're testing
        optimize_prompt(self.initial_prompt, self.output_format_prompt, self.eval_data, self.iterations)

        # Assert that the evaluate_prompt function was called the correct number of times
        self.assertEqual(mock_evaluate_prompt.call_count, self.iterations)

        # Assert that the generate_new_prompt function was called the correct number of times
        self.assertEqual(mock_generate_new_prompt.call_count, self.iterations - 1)

        # Assert that the display functions were called
        mock_display_metrics.assert_called()
        mock_display_best_prompt.assert_called()
        mock_display_comparison_table.assert_called()

        # Assert that the log_final_results function was called
        mock_log_final_results.assert_called()

    def test_estimate_token_usage(self):
        prompt = "This is a test prompt"
        output_format_prompt = "Test output format"
        eval_data = pd.DataFrame({'text': ['This is a test', 'Another test']})
        iterations = 3

        estimated_tokens = estimate_token_usage(prompt, output_format_prompt, eval_data, iterations)

        # Assert that the estimated token usage is greater than zero
        self.assertGreater(estimated_tokens, 0)

    @patch('src.iterative_prompt_optimization.utils.PRICING', {'test_provider': {'test_model': 0.01}})
    @patch('src.iterative_prompt_optimization.utils.SELECTED_PROVIDER', 'test_provider')
    @patch('src.iterative_prompt_optimization.utils.MODEL_NAME', 'test_model')
    def test_estimate_cost(self):
        token_usage = 1000
        provider = 'test_provider'
        model = 'test_model'
        estimated_cost = estimate_cost(token_usage, provider, model)
        self.assertEqual(estimated_cost, '$0.01')

    @patch('src.iterative_prompt_optimization.utils.Console')
    def test_display_best_prompt(self, mock_console):
        best_prompt = "Best prompt"
        output_format_prompt = "Output format"

        display_best_prompt(best_prompt, output_format_prompt)

        # Assert that the console's print method was called
        mock_console.return_value.print.assert_called()

    @patch('src.iterative_prompt_optimization.utils.Console')
    def test_display_comparison_table(self, mock_console):
        all_metrics = [
            {'iteration': 1, 'precision': 0.8, 'recall': 0.7, 'accuracy': 0.75, 'f1': 0.74, 'invalid_predictions': 0},
            {'iteration': 2, 'precision': 0.85, 'recall': 0.75, 'accuracy': 0.8, 'f1': 0.79, 'invalid_predictions': 0}
        ]

        display_comparison_table(all_metrics)

        # Assert that the console's print method was called
        mock_console.return_value.print.assert_called()

    def test_create_metric_entry(self):
        iteration = 1
        results = {
            'precision': 0.8,
            'recall': 0.7,
            'accuracy': 0.75,
            'f1': 0.74,
            'invalid_predictions': 0
        }

        metric_entry = create_metric_entry(iteration, results)

        # Assert that the metric entry contains all the expected keys
        self.assertSetEqual(set(metric_entry.keys()), 
                            {'iteration', 'precision', 'recall', 'accuracy', 'f1', 'invalid_predictions'})
        # Assert that the values are correct
        self.assertEqual(metric_entry['iteration'], iteration)
        self.assertEqual(metric_entry['precision'], results['precision'])

    def test_update_best_metrics(self):
        best_metrics = {
            'precision': 0.8,
            'recall': 0.7,
            'accuracy': 0.75,
            'f1': 0.74,
            'invalid_predictions': 0
        }
        best_prompt = "Best prompt"
        current_prompt = "Current prompt"

        # Test when current results are better
        better_results = {
            'precision': 0.85,
            'recall': 0.75,
            'accuracy': 0.8,
            'f1': 0.79,
            'invalid_predictions': 0
        }
        new_best_metrics, new_best_prompt = update_best_metrics(best_metrics, best_prompt, better_results, current_prompt)
        self.assertEqual(new_best_metrics, better_results)
        self.assertEqual(new_best_prompt, current_prompt)

        # Test when current results are worse
        worse_results = {
            'precision': 0.75,
            'recall': 0.65,
            'accuracy': 0.7,
            'f1': 0.69,
            'invalid_predictions': 1
        }
        new_best_metrics, new_best_prompt = update_best_metrics(best_metrics, best_prompt, worse_results, current_prompt)
        self.assertEqual(new_best_metrics, best_metrics)
        self.assertEqual(new_best_prompt, best_prompt)

if __name__ == '__main__':
    unittest.main()