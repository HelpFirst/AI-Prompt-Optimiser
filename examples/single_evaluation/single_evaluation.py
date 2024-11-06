from iterative_prompt_optimization.evaluation import single_prompt_evaluation
import pandas as pd

def main():
    # Example evaluation dataset
    eval_data = pd.DataFrame({
        'text': [
            'This product is amazing! Best purchase ever.',
            'Terrible quality, broke after one use.',
            'Pretty good but a bit expensive.'
        ],
        'label': [1, 0, 1]  # 1 for positive, 0 for negative
    })

    # Output schema for binary classification
    output_schema = {
        'key_to_extract': 'sentiment',
        'value_mapping': {
            'positive': 1,
            'negative': 0
        },
        'regex_pattern': r'"sentiment":\s*"(.*?)"',
        'chain_of_thought_key': 'reasoning',
        'chain_of_thought_regex': r'"reasoning":\s*"(.*?)"'
    }

    # Example prompt
    prompt = """Analyze the sentiment of the given text and classify it as positive or negative.

Please think through this step by step:
1. Identify key sentiment words and phrases
2. Consider the overall context
3. Determine if the sentiment is predominantly positive or negative

Output your analysis in the following format:
{
    "sentiment": "positive/negative",
    "reasoning": "your step-by-step reasoning here"
}

Text to analyze:
"""

    # Run evaluation
    results = single_prompt_evaluation(
        prompt=prompt,
        eval_data=eval_data,
        output_schema=output_schema,
        eval_provider="openai",
        eval_model="gpt-3.5-turbo",
        eval_temperature=0.7,
        experiment_name="sentiment_analysis_test"  # Optional: provide name to generate dashboard
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Valid Predictions: {results['valid_predictions']}")
    print(f"Invalid Predictions: {results['invalid_predictions']}")

if __name__ == "__main__":
    main() 