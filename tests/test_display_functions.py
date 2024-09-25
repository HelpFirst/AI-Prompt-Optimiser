import unittest
from unittest.mock import patch, MagicMock
from src.iterative_prompt_optimization.utils import display_best_prompt, display_comparison_table

class TestDisplayFunctions(unittest.TestCase):
    @patch('src.iterative_prompt_optimization.utils.Console')
    def test_display_best_prompt(self, mock_console):
        best_prompt = "Best prompt"
        output_format_prompt = "Output format"

        display_best_prompt(best_prompt, output_format_prompt)

        # Assert that the console's print method was called
        mock_console.return_value.print.assert_called()

        # Check if the print method was called with the correct arguments
        calls = mock_console.return_value.print.call_args_list
        full_output = ' '.join(str(call) for call in calls)
        self.assertIn('Best prompt', full_output)
        self.assertIn('Output format', full_output)

    @patch('src.iterative_prompt_optimization.utils.Console')
    def test_display_comparison_table(self, mock_console):
        all_metrics = [
            {'iteration': 1, 'precision': 0.8, 'recall': 0.7, 'accuracy': 0.75, 'f1': 0.74, 'invalid_predictions': 0},
            {'iteration': 2, 'precision': 0.85, 'recall': 0.75, 'accuracy': 0.8, 'f1': 0.79, 'invalid_predictions': 0}
        ]

        display_comparison_table(all_metrics)

        # Assert that the console's print method was called
        mock_console.return_value.print.assert_called()

        # Check if the print method was called with a Table object
        calls = mock_console.return_value.print.call_args_list
        self.assertTrue(any('Table' in str(call) for call in calls))

if __name__ == '__main__':
    unittest.main()