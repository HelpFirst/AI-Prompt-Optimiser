import unittest
from src.iterative_prompt_optimization.utils import create_metric_entry, update_best_metrics

class TestMetricFunctions(unittest.TestCase):
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
        self.assertEqual(metric_entry['recall'], results['recall'])
        self.assertEqual(metric_entry['accuracy'], results['accuracy'])
        self.assertEqual(metric_entry['f1'], results['f1'])
        self.assertEqual(metric_entry['invalid_predictions'], results['invalid_predictions'])

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

        # Test when current results are equal
        equal_results = best_metrics.copy()
        new_best_metrics, new_best_prompt = update_best_metrics(best_metrics, best_prompt, equal_results, current_prompt)
        self.assertEqual(new_best_metrics, best_metrics)
        self.assertEqual(new_best_prompt, best_prompt)

if __name__ == '__main__':
    unittest.main()